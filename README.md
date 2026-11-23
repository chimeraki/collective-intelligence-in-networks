# Information Placement and Collective Inference in Networks

Code and simulated data accompanying *Information Placement and Collective Inference in Networks*.

**Author:** Sanjukta Krishnagopal  
**Affiliation:** Department of Computer Science, University of California, Santa Barbara.

This repository contains the **reproduction code and data only**. The manuscript, its figures, and LaTeX source are intentionally maintained separately.

## Files

- `src/`: simulation, analysis, and optimization code
- `data/`: archived simulated observations and results
- `rerun/`: outputs from the recorded reproducibility audit
- `RUN_EXPERIMENTS.sh`: entry point for reproducing experiments
- `verify_archives.py`: verifies deterministic results
- `REPRODUCIBILITY_AUDIT.md`: documentation of checks and limitations

## Run

Requires Python 3.11 or later.

```bash
python -m pip install -r requirements.txt
bash RUN_EXPERIMENTS.sh
```

The primary 96-graph matched experiment, exact DGD risk, rank bounds, higher-order controls, and fixed-proposal design tests were reproduced in the documented environment. The separate 48-graph fresh-seed replication has archived data, but its original generating script is unavailable. Runtime-budget experiments can vary across machines. See `REPRODUCIBILITY_AUDIT.md`.

## Citation

Sanjukta Krishnagopal, *Information Placement and Collective Inference in Networks*, manuscript.

## License

See `LICENSE`.
