# Codificación del estado de juego

Este documento explica cómo se transforma cada decisión de Pluribus en un vector numérico
para la red neuronal. El código está en `src/data_loader.py` (extracción del estado desde los
logs) y `src/state_encoder.py` (conversión del estado en variables).

Se construyen dos codificaciones:

| Modelo | Nº variables | Idea |
|---|---|---|
| **A — Observación** (referencia) | 20 | Solo lo que se ve en la mesa, sin conceptos de póker |
| **C — Feature engineering** (definitivo) | 20 | Las cartas se resumen con conocimiento de póker |

Ambas comparten exactamente las mismas 9 variables de **contexto de apuestas**; solo
cambia cómo se representan las cartas. Por eso la diferencia de rendimiento entre A y C
mide cuánto aporta el conocimiento de póker sobre las cartas.

---

## 1. Principios de diseño

1. **Solo información que Pluribus tiene al decidir.** Sus dos cartas, el board y lo
   ocurrido en la mesa hasta ese momento. Los logs incluyen las cartas de los rivales
   (`Dealt to MrWhite [...]`), pero no se usan nunca.
2. **Estado anterior a la acción.** Cada registro se guarda *antes* de aplicar la acción
   de Pluribus al bote, a las fichas y a los jugadores activos. Así ninguna variable
   contiene información de la etiqueta que se quiere predecir (fuga de datos).
3. **Normalización con constantes del juego, no del dataset.** Ciega grande = 100,
   stacks iniciales = 10000, bote máximo = 600 BB. Ninguna escala se calcula con los
   datos, así que el conjunto de test no influye en el de entrenamiento. Todas las
   variables quedan en [0, 1].
4. **Simetría de palos.** En póker los palos concretos no importan (A♠K♠ ≡ A♥K♥); solo
   importan las relaciones entre ellos (mismo palo o no). Ninguna variable codifica un
   palo concreto.
5. **Pocas variables.** Con ~12000 decisiones de entrenamiento, cada variable tiene que
   justificar su presencia. Las cartas en bruto (7 cartas × 5 valores = 35 variables,
   dependientes del orden) se sustituyeron por resúmenes compactos.

---

## 2. Extracción del estado (`data_loader.py`)

Para cada mano se reconstruye el estado de la mesa línea a línea:

| Aspecto | Cómo se obtiene |
|---|---|
| Ciegas | Líneas `posts small/big blind`: se suman al bote y a las fichas puestas por cada jugador. El bote preflop empieza en 150. |
| Apuestas | `bets X` y `calls X`: X es lo que se añade. `raises X to Y`: Y es el total de la calle, así que se añade Y − lo ya puesto en la calle. |
| Fichas por calle | Se reinician en cada calle (flop, turn, river). |
| Board | Se leen todas las cartas de la línea; en turn y river la carta nueva viene en un segundo corchete (`[5c 9s 7c] [3h]`). |
| Cantidad a pagar | Apuesta más alta de la calle − fichas que Pluribus ya ha puesto en la calle, con un tope igual a su stack restante (all-in). |
| Orden de actuación | Preflop empieza el jugador a la izquierda de la ciega grande; postflop, el siguiente al botón. |
| Jugadores all-in | Se marcan para no contarlos como jugadores que aún deciden. |
| Sesión | Se lee de la cabecera (`Pluribus Session 101`); sirve para separar train y test por sesión. |

Comprobación de coherencia: en las 15169 decisiones no hay ninguna acción imposible (un
*check* teniendo que pagar, un *call* o un *fold* sin nada que pagar, o un *bet* frente a
una apuesta). Eso confirma que la cantidad a pagar se reconstruye bien.

**Etiqueta (y):** la acción de Pluribus: `fold`, `check`, `call`, `bet` o `raise`.

---

## 3. Contexto de apuestas (común a A y C, 9 variables)

| Variable | Fórmula | Qué representa |
|---|---|---|
| `street` | calle / 3 | 0 = preflop, 0.33 = flop, 0.67 = turn, 1 = river. |
| `position` | ((asiento − botón − 1) mod 6) / 5 | 0 = SB, 0.2 = BB, 0.4 = UTG, 0.6 = MP, 0.8 = CO, 1 = BTN. |
| `players_active` | (jugadores sin foldear − 2) / 4 | Cuántos siguen en la mano (2 a 6), Pluribus incluido. |
| `players_behind` | rivales que hablan después / 5 | Rivales activos (sin foldear ni all-in) que aún actúan después de Pluribus en esta calle. Mide la posición *real* en la calle: 0 = cierra la acción. |
| `pot_size` | log(1 + bote/BB) / log(1 + 600) | Tamaño del bote en ciegas grandes, en escala logarítmica (de 1.5 BB a cientos de BB). |
| `pot_odds` | a_pagar / (bote + a_pagar) | Proporción del bote final que pone Pluribus si paga. 0 = no tiene que pagar nada (puede pasar). Recoge a la vez si se enfrenta a una apuesta y su tamaño relativo. |
| `spr` | log(1 + stack_efectivo/bote) / log(1 + 100) | Stack-to-pot ratio. Stack efectivo = mín(stack de Pluribus, mayor stack de un rival activo): lo máximo que realmente puede ganar o perder. |
| `num_raises` | mín(apuestas + subidas en la calle, 4) / 4 | Nivel de agresión de la calle: 0 = nadie ha apostado (las ciegas no cuentan), 0.25 = una apuesta o subida inicial, 0.5 = una resubida (3-bet)... |
| `aggressor_prev_street` | 0 / 1 | Si Pluribus hizo la última apuesta o subida de la calle anterior (p. ej. subió preflop y llega al flop con la iniciativa). Siempre 0 en preflop. |

---

## 4. Cartas — Modelo A (observación, 11 variables)

Solo rangos y relaciones de palo, sin interpretar la jugada.

| Variable | Fórmula | Qué representa |
|---|---|---|
| `hole_high` | rango / 12 | Carta propia más alta (2 = 0, A = 1). |
| `hole_low` | rango / 12 | Carta propia más baja. |
| `hole_suited` | 0 / 1 | Las dos cartas propias son del mismo palo. |
| `board_rank_1` … `board_rank_5` | (rango + 1) / 13 | Cartas del board ordenadas de mayor a menor (2 ≈ 0.08, A = 1). 0 = carta aún no salida. Se ordenan porque el orden de salida no cambia la jugada. |
| `board_suit_match_high` | nº / 5 | Cartas del board del mismo palo que la carta propia alta. |
| `board_suit_match_low` | nº / 5 | Cartas del board del mismo palo que la carta propia baja. |
| `board_max_suit` | nº / 5 | Máximo de cartas del board de un mismo palo. |

Las parejas, los proyectos o la fuerza de la mano **no** se dan: el modelo tiene que
deducirlos comparando rangos y palos.

---

## 5. Cartas — Modelo C (feature engineering, 11 variables)

Las cartas se resumen en los conceptos que usa un jugador de póker.

### 5.1 Cartas propias (4)

| Variable | Fórmula | Qué representa |
|---|---|---|
| `hole_high` | rango / 12 | Carta propia más alta. |
| `hole_low` | rango / 12 | Carta propia más baja. |
| `hole_pair` | 0 / 1 | Pareja servida. |
| `hole_suited` | 0 / 1 | Mismo palo. |

Con estas cuatro variables se distinguen las 169 manos iniciales distintas.

### 5.2 Fuerza de la mano (5)

| Variable | Fórmula | Qué representa |
|---|---|---|
| `hand_category` | categoría / 8 | Mejor jugada con cartas propias + board: 0 carta alta, 1 pareja, 2 dobles parejas, 3 trío, 4 escalera, 5 color, 6 full, 7 póker, 8 escalera de color. En preflop: 1/8 si es pareja servida, 0 si no. |
| `board_category` | categoría / 8 | Jugada que forma el board solo, es decir, la que comparten todos los jugadores. Junto con `hand_category` indica si las cartas propias mejoran el board (p. ej. "dobles parejas" con el board doblado vale mucho menos). 0 en preflop. |
| `pair_strength` | fracción del board por debajo de la pareja | Solo si la jugada es una pareja con al menos una carta propia: 1 = overpair, 0.67–0.8 = top pair, valores intermedios = pareja media, 0 = pareja por debajo del board o pareja solo en el board. |
| `flush_draw` | 0 / 1 | Proyecto de color: exactamente 4 cartas de un palo con al menos una propia. Solo en flop y turn. |
| `straight_draw` | 0 / 0.5 / 1 | Proyecto de escalera: número de rangos que completarían una escalera con las cartas propias (y no con el board solo). 0.5 = gutshot (1 rango), 1 = abierto o doble gutshot (2 rangos). Solo en flop y turn, y 0 si ya hay escalera. |

### 5.3 Textura del board (2)

| Variable | Fórmula | Qué representa |
|---|---|---|
| `board_flush_possible` | 0 / 1 | 3 o más cartas del board del mismo palo: algún rival puede tener color. |
| `board_straight_possible` | 0 / 1 | 3 o más rangos distintos del board caben en una ventana de 5 (contando el As como 1): algún rival puede tener escalera. |

---

## 6. Qué no se codifica y por qué

| Aspecto | Motivo |
|---|---|
| Cartas de los rivales | Pluribus no las ve al decidir; usarlas sería fuga de información. |
| Identidad y estadísticas de los rivales | Pluribus juega una estrategia fija que no se adapta a cada rival. |
| Historial completo de la mano | Se resume en `num_raises`, `pot_odds`, `pot_size` y `aggressor_prev_street`. Probar más historial no mejoró el modelo (ver §7). |
| Stacks absolutos | Todas las manos empiezan con 10000; lo relevante es el stack relativo al bote (`spr`). |
| Tamaño de las apuestas de Pluribus | Es parte de su decisión, no del estado; podría ser un segundo objetivo de predicción. |
| Equity Monte Carlo | Se probó (modelo B) y no mejoró a C: resume la misma información que las variables de fuerza de mano (ver §7). |

---

## 7. Validación de las decisiones de diseño

Todas las variantes se compararon con validación cruzada de 5 particiones agrupada por
sesión, **solo con las sesiones de entrenamiento** (el test no se usó para elegir), y con
la media de 3 semillas de la red.

| Variante | Nº vars | Accuracy | Macro F1 | Log loss |
|---|---|---|---|---|
| **C (definitivo)** | 20 | **0.852** | 0.753 | **0.328** |
| C sin las 3 variables de menor importancia¹ | 17 | 0.846 | 0.742 | 0.341 |
| C + Pluribus fue agresor preflop | 21 | 0.848 | 0.747 | 0.336 |
| C + Pluribus ya apostó/subió en esta calle | 21 | 0.849 | 0.749 | 0.335 |
| C + ambas | 22 | 0.848 | 0.746 | 0.338 |
| C + ambas − 3 de menor importancia | 19 | 0.851 | 0.755 | 0.328 |

¹ `board_flush_possible`, `pair_strength`, `board_straight_possible`. Por separado tienen
poca importancia por permutación, pero quitarlas juntas empeora el modelo: aportan
información complementaria.

Modelo B (C + equity Monte Carlo contra rivales aleatorios), en test: 0.847 de accuracy
frente a 0.846 de C, y 0.849 frente a 0.848 en validación cruzada. Sin diferencia
significativa, así que se descartó y su código no forma parte de la versión final.

---

## 8. Resultados finales (test: 14 sesiones no vistas, 3228 decisiones)

| | Clase mayoritaria | Mayoritaria por calle | A (observación) | **C (definitivo)** |
|---|---|---|---|---|
| Accuracy | 0.526 | 0.640 | 0.822 | **0.846** |
| Balanced accuracy | 0.200 | 0.379 | 0.721 | **0.727** |
| Macro F1 | 0.138 | 0.290 | 0.720 | **0.737** |
| Log loss | – | – | 0.357 | **0.341** |
| Preflop | | | 0.917 | 0.923 |
| Flop | | | 0.668 | 0.721 |
| Turn | | | 0.627 | 0.711 |
| River | | | 0.548 | 0.592 |

- **Preflop** los dos modelos rinden igual: con 169 manos posibles, la red aprende sola
  a partir de los rangos.
- **Postflop** el conocimiento de póker aporta entre 4 y 8 puntos por calle: las
  combinaciones de cartas son demasiadas para aprenderlas con ~12000 ejemplos.
- La acción más difícil es `bet` (recall ≈ 0.33): Pluribus juega una estrategia mixta,
  así que en situaciones idénticas a veces apuesta y a veces pasa. Parte de ese error es
  irreducible.
