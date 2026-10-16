# backstop

M0 and M0b tested whether a locked confidence threshold needs a teacher-labeled per-slice loop. Both pre-registered results are **KILL**. Do not build the runtime on this evidence.

- M0: drift that moves the input also moves confidence and coverage. [docs/M0-results.md](docs/M0-results.md)
- M0b: a label change on the same text hides from those monitors, and a 5% teacher sample plus auto-refit does not restore the target. [docs/M0b-results.md](docs/M0b-results.md)

Prior art: [docs/prior-art.md](docs/prior-art.md). Frozen rules: [docs/M0-predictions.md](docs/M0-predictions.md), [docs/M0b-predictions.md](docs/M0b-predictions.md).

```bash
pip install -r requirements.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
make m0
make m0b
```
