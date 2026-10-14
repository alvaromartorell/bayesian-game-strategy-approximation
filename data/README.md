# Data

The model is trained on the public hand histories of Pluribus: 10,000 hands of six-player
No-Limit Hold'em against professional players, split across 92 log files (67 sessions).

The logs are **not included** in this repository because they come from a third party.
You can download them from:

- Kevin Wang, *Let's Analyze Pluribus's Hands*:
  https://www.kevinwang.us/blog/lets-analyze-pluribuss-hands/

Put the `.txt` files in `data/pluribus_logs/`:

```
data/
└── pluribus_logs/
    ├── pluribus_40.txt
    ├── pluribus_40b.txt  (some sessions are split in two files)
    └── ...
```

The logs use the PokerStars text format and show every player's hole cards. The code never
uses the opponents' cards: Pluribus did not see them when it acted.
