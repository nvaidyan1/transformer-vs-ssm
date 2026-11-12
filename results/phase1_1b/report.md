## Part 1: fixed-position Selective Copying (seed 0, CAP 3000)

| arm | steps-to-0.95 | steps run | early stop | final val_acc |
|---|---|---|---|---|
| ablated | not reached | 3000 | False | 0.2578 |
| selective | 1250 | 2250 | True | 1.0000 |

Gate (ablated >= 0.9): **FAIL**

## Part 2: the 1.1 ablated run (re-run with the identical command)

- Final val_acc 0.0205 (original 1.1 run: 0.0205); 1/62 = 0.0161, the trainer's chance_acc = 1/31 = 0.0323.
- Final training loss 4.142 (last logged batch) vs ln(31) = 3.434 and ln(62) = 4.127.
- Predictions: 29 distinct tokens over 1024 answers (62 distinct targets); 0 non-data predictions; entropy 1.393 nats (uniform over 62 = 4.127); top 5 (id, count): [(42, 732), (53, 75), (4, 19), (18, 18), (26, 16)].
