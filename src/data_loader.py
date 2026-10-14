import re
from glob import glob
import pandas as pd

# Expresiones regulares
HAND_START = re.compile(r"^PokerStars Hand #(\d+):")
TABLE_LINE = re.compile(r"^Table 'Pluribus Session (\w+)'.*Seat #(\d+) is the button")
SEAT_LINE = re.compile(r"^Seat (\d+): (\S+) \((\d+) in chips\)")
BLIND_LINE = re.compile(r"^(\S+): posts (small|big) blind (\d+)")
DEALT_LINE = re.compile(r"^Dealt to Pluribus \[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]")
ACTION_LINE = re.compile(r"^(\S+): (folds|checks|calls (\d+)|bets (\d+)|raises \d+ to (\d+))")
BOARD_LINE = re.compile(r"^\*\*\* (FLOP|TURN|RIVER) \*\*\*(.*)$")
CARD = re.compile(r"[2-9TJQKA][shdc]")
SUMMARY_LINE = re.compile(r"^\*\*\* SUMMARY \*\*\*")

HERO = 'Pluribus'


def parse_logs(path_pattern):
    hands = []
    for path in sorted(glob(path_pattern)):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_hand = []
        for line in lines:
            if HAND_START.match(line):
                if current_hand:
                    hands.append(current_hand)
                current_hand = [line.strip()]
            elif current_hand:
                current_hand.append(line.strip())
        if current_hand:
            hands.append(current_hand)

    print(f"Total manos detectadas: {len(hands)}")
    records = [r for hand in hands for r in _extract_pluribus_decisions(hand)]
    print(f"Decisiones de Pluribus extraídas: {len(records)}")
    return pd.DataFrame(records)


def _extract_pluribus_decisions(hand):
    """
    Recorre una mano y devuelve un registro por cada decisión de Pluribus con el estado
    del juego JUSTO ANTES de su acción. El estado se actualiza con la acción después de
    guardar el registro, para que ninguna variable contenga información de la etiqueta.
    """
    hand_id = HAND_START.match(hand[0]).group(1)
    session, button = None, None
    start_stack, seat_of = {}, {}
    street_commit, total_commit = {}, {}   # fichas puestas en esta calle / en toda la mano
    folded, all_in = set(), set()
    hole = None
    board = []
    street = 0
    pot = 0
    current_bet = 0
    num_raises = 0
    last_aggressor = None
    prev_street_aggressor = None
    bb_player = None
    records = []

    def acting_order(player):
        # Preflop habla primero el jugador tras la ciega grande; postflop, el siguiente al botón
        first_after = seat_of[bb_player] if street == 0 and bb_player else button
        return (seat_of[player] - first_after - 1) % 10

    for line in hand[1:]:
        m = TABLE_LINE.match(line)
        if m:
            session, button = m.group(1), int(m.group(2))
            continue

        m = SEAT_LINE.match(line)
        if m:
            seat, name, stack = m.groups()
            seat_of[name] = int(seat)
            start_stack[name] = int(stack)
            street_commit[name] = 0
            total_commit[name] = 0
            continue

        m = BLIND_LINE.match(line)
        if m:
            name, kind, amount = m.group(1), m.group(2), int(m.group(3))
            street_commit[name] += amount
            total_commit[name] += amount
            pot += amount
            current_bet = max(current_bet, amount)
            if kind == 'big':
                bb_player = name
            continue

        m = DEALT_LINE.match(line)
        if m:
            hole = m.group(1), m.group(2)
            continue

        m = BOARD_LINE.match(line)
        if m:
            board = CARD.findall(m.group(2))   # el turn y el river vienen en un segundo corchete
            street += 1
            prev_street_aggressor = last_aggressor
            street_commit = dict.fromkeys(street_commit, 0)
            current_bet = 0
            num_raises = 0
            last_aggressor = None
            continue

        if SUMMARY_LINE.match(line):
            break

        m = ACTION_LINE.match(line)
        if not m or m.group(1) not in seat_of:
            continue
        actor, action_raw = m.group(1), m.group(2)
        action = action_raw.split()[0].rstrip('s')   # folds -> fold, raises -> raise, ...

        # --- Guardar el estado ANTES de aplicar la acción ---
        if actor == HERO and hole is not None:
            remaining = {p: start_stack[p] - total_commit[p] for p in seat_of}
            opponents = [p for p in seat_of if p != HERO and p not in folded]
            behind = [p for p in opponents
                      if p not in all_in and acting_order(p) > acting_order(HERO)]
            records.append({
                'session': session,
                'hand_id': hand_id,
                'hole1': hole[0],
                'hole2': hole[1],
                'board': list(board),
                'street': street,                                    # 0=preflop ... 3=river
                'position': (seat_of[HERO] - button - 1) % len(seat_of),  # 0=SB ... 5=BTN
                'pot': pot,
                'to_call': min(max(current_bet - street_commit[HERO], 0), remaining[HERO]),  # all-in: paga lo que tiene
                'stack': remaining[HERO],
                'eff_stack': min(remaining[HERO], max((remaining[p] for p in opponents), default=0)),
                'num_players_active': len(opponents) + 1,
                'players_behind': len(behind),
                'num_raises': num_raises,
                'aggressor_prev_street': int(prev_street_aggressor == HERO),
                'action': action,
            })

        # --- Aplicar la acción al estado ---
        if action == 'fold':
            folded.add(actor)
            continue
        if action == 'check':
            continue
        if action == 'call':
            amount = int(m.group(3))
        elif action == 'bet':
            amount = int(m.group(4))
        else:  # raise: "raises X to Y" -> Y es el total de la calle, no lo que se añade
            amount = int(m.group(5)) - street_commit[actor]
        street_commit[actor] += amount
        total_commit[actor] += amount
        pot += amount
        if action in ('bet', 'raise'):
            current_bet = street_commit[actor]
            num_raises += 1
            last_aggressor = actor
        if total_commit[actor] >= start_stack[actor]:
            all_in.add(actor)

    return records
