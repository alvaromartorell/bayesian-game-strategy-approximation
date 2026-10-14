#encodingestados5
"""
Dos codificaciones del estado de juego desde el punto de vista de Pluribus
(explicación detallada en CODIFICACION.md):

  A) OBSERVACIÓN: solo lo que se ve en la mesa (rangos de las cartas, relaciones de palo
     y contexto de apuestas). Ningún concepto de póker (parejas, proyectos...).
     Los palos se codifican de forma relativa porque los palos concretos son simétricos:
     A♠K♠ y A♥K♥ son la misma mano.

  C) FEATURE ENGINEERING (modelo definitivo): mismo contexto de apuestas, pero las cartas
     se resumen con conocimiento de póker: categoría de mano, proyectos y textura del board.

Las dos comparten exactamente las mismas variables de contexto, así que la diferencia
entre los modelos A y C mide cuánto aporta el conocimiento de póker sobre las cartas.

Todas las normalizaciones usan constantes fijas del juego (ciegas 50/100, stacks de
10000), no estadísticas del dataset, para no filtrar información del test al train.
"""
from collections import Counter
import numpy as np

RANKS = '23456789TJQKA'
BIG_BLIND = 100
MAX_POT_BB = 600          # 6 jugadores x 100 BB
MAX_SPR = 100

CONTEXT_FEATURES = [
    'street',               # 0=preflop ... 1=river
    'position',             # 0=SB ... 1=BTN
    'players_active',       # jugadores que siguen en la mano (2-6)
    'players_behind',       # rivales que aún actúan después de Pluribus en esta calle
    'pot_size',             # bote en ciegas grandes, escala log
    'pot_odds',             # to_call / (bote + to_call); 0 si no hay que pagar
    'spr',                  # stack efectivo / bote, escala log
    'num_raises',           # apuestas/subidas voluntarias en esta calle
    'aggressor_prev_street',  # Pluribus fue el último agresor de la calle anterior
]

FEATURE_NAMES_OBS = [
    'hole_high',            # rango de la carta propia alta (0-1)
    'hole_low',             # rango de la carta propia baja (0-1)
    'hole_suited',          # las dos cartas propias son del mismo palo
    'board_rank_1', 'board_rank_2', 'board_rank_3', 'board_rank_4', 'board_rank_5',
                            # rangos del board ordenados de mayor a menor (0 = carta aún no salida)
    'board_suit_match_high',  # cartas del board con el palo de la carta propia alta (/5)
    'board_suit_match_low',   # cartas del board con el palo de la carta propia baja (/5)
    'board_max_suit',       # nº máximo de cartas del board de un mismo palo (/5)
] + CONTEXT_FEATURES

FEATURE_NAMES_ENG = [
    # --- Cartas privadas (4) ---
    'hole_high',            # rango de la carta alta (0-1)
    'hole_low',             # rango de la carta baja (0-1)
    'hole_pair',            # pareja servida
    'hole_suited',          # mismo palo
    # --- Fuerza de la mano (5) ---
    'hand_category',        # mejor jugada con hole+board: 0=carta alta ... 1=escalera de color
    'board_category',       # jugada que da el board solo (la que comparten todos)
    'pair_strength',        # si la jugada es pareja con carta propia: 1=overpair, ~0.67-1=top pair, 0=bajo board
    'flush_draw',           # 4 cartas del mismo palo con al menos una propia (flop/turn)
    'straight_draw',        # 0, 0.5=gutshot, 1=abierto (flop/turn)
    # --- Textura del board (2) ---
    'board_flush_possible', # >=3 cartas del mismo palo en el board
    'board_straight_possible',  # 3 cartas del board caben en una escalera
] + CONTEXT_FEATURES


def _rank(card):
    return RANKS.index(card[0])


def _has_straight(ranks):
    r = set(ranks)
    if 12 in r:
        r.add(-1)  # el As también cuenta como 1
    return any(all(x + i in r for i in range(5)) for x in range(-1, 9))


def hand_category(cards):
    """Categoría de la mejor jugada: 0=carta alta, 1=pareja, 2=dobles, 3=trío, 4=escalera,
    5=color, 6=full, 7=póker, 8=escalera de color. Funciona con 2 a 7 cartas."""
    if not cards:
        return 0
    ranks = [_rank(c) for c in cards]
    suits = [c[1] for c in cards]
    counts = sorted(Counter(ranks).values(), reverse=True) + [0]
    flush_suit = next((s for s, n in Counter(suits).items() if n >= 5), None)
    if flush_suit and _has_straight([_rank(c) for c in cards if c[1] == flush_suit]):
        return 8
    if counts[0] == 4:
        return 7
    if counts[0] == 3 and counts[1] >= 2:
        return 6
    if flush_suit:
        return 5
    if _has_straight(ranks):
        return 4
    if counts[0] == 3:
        return 3
    if counts[0] == 2 and counts[1] == 2:
        return 2
    if counts[0] == 2:
        return 1
    return 0


def _pair_strength(hole, board):
    """Para una pareja hecha con al menos una carta propia, qué parte del board queda por debajo."""
    hole_r = [_rank(c) for c in hole]
    board_r = sorted({_rank(c) for c in board})
    if hole_r[0] == hole_r[1]:
        pair = hole_r[0]
    else:
        matches = [r for r in hole_r if r in board_r]
        if not matches:
            return 0.0  # la pareja está solo en el board
        pair = max(matches)
    return sum(b < pair for b in board_r) / len(board_r)


def _draws(hole, board):
    """Proyectos de color y escalera (solo con cartas por salir: flop y turn)."""
    if len(board) not in (3, 4):
        return 0.0, 0.0
    cards = list(hole) + list(board)
    suit_counts = Counter(c[1] for c in cards)
    flush_draw = float(any(n == 4 and any(h[1] == s for h in hole) for s, n in suit_counts.items()))

    ranks = {_rank(c) for c in cards}
    if _has_straight(ranks):
        return flush_draw, 0.0
    outs = sum(_has_straight(ranks | {k}) and not _has_straight({_rank(c) for c in board} | {k})
               for k in range(13) if k not in ranks)
    return flush_draw, min(outs, 2) / 2


def _board_texture(board):
    if len(board) < 3:
        return 0.0, 0.0
    flush_possible = float(max(Counter(c[1] for c in board).values()) >= 3)
    r = {_rank(c) for c in board}
    if 12 in r:
        r.add(-1)
    straight_possible = float(any(sum(x + i in r for i in range(5)) >= 3 for x in range(-1, 9)))
    return flush_possible, straight_possible


def _sorted_hole(state):
    return sorted([state['hole1'], state['hole2']], key=_rank, reverse=True)


def _context_features(state):
    pot = max(state['pot'], 1)
    to_call = state['to_call']
    return [
        state['street'] / 3,
        state['position'] / 5,
        (state['num_players_active'] - 2) / 4,
        state['players_behind'] / 5,
        min(np.log1p(pot / BIG_BLIND) / np.log1p(MAX_POT_BB), 1.0),
        to_call / (pot + to_call),
        min(np.log1p(state['eff_stack'] / pot) / np.log1p(MAX_SPR), 1.0),
        min(state['num_raises'], 4) / 4,
        float(state['aggressor_prev_street']),
    ]


def encode_observation(state):
    """Modelo A: vector de len(FEATURE_NAMES_OBS) valores en [0, 1]."""
    high, low = _sorted_hole(state)
    board = list(state['board'])

    board_ranks = sorted((_rank(c) + 1 for c in board), reverse=True)
    board_ranks = [r / 13 for r in board_ranks] + [0.0] * (5 - len(board_ranks))
    board_suits = Counter(c[1] for c in board)

    card_feats = [
        _rank(high) / 12,
        _rank(low) / 12,
        float(high[1] == low[1]),
        *board_ranks,
        board_suits[high[1]] / 5,
        board_suits[low[1]] / 5,
        max(board_suits.values(), default=0) / 5,
    ]
    return np.array(card_feats + _context_features(state), dtype=float)


def encode_engineered(state):
    """Modelo C: vector de len(FEATURE_NAMES_ENG) valores en [0, 1]."""
    hole = _sorted_hole(state)
    board = list(state['board'])

    hi, lo = _rank(hole[0]), _rank(hole[1])
    hole_feats = [hi / 12, lo / 12, float(hi == lo), float(hole[0][1] == hole[1][1])]

    if board:
        cat = hand_category(hole + board)
        board_cat = hand_category(board)
        pair_str = _pair_strength(hole, board) if cat == 1 else 0.0
        flush_draw, straight_draw = _draws(hole, board)
    else:
        cat = hand_category(hole)  # preflop: 1 si es pareja, 0 si no
        board_cat = 0
        pair_str = flush_draw = straight_draw = 0.0
    strength_feats = [cat / 8, board_cat / 8, pair_str, flush_draw, straight_draw]

    texture_feats = list(_board_texture(board))

    return np.array(hole_feats + strength_feats + texture_feats + _context_features(state), dtype=float)
