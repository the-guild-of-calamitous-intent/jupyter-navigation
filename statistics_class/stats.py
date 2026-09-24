import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Statistics
    """)
    return


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt

    return (np,)


@app.cell
def _(np):
    class Stats:
        def __init__(self):
            self.cnt = 0
            self.m = 0
            self.v = 0
        def update(self, val):
            diff = val - self.m
            self.cnt += 1
            self.m += diff / self.cnt
            self.v += diff*diff
        def mean(self): 
            return self.m
        def var(self): 
            if self.v <= 1: return 0
            return self.v / (self.cnt - 1)
        def std(self): 
            return np.sqrt(self.var())

    return (Stats,)


@app.cell
def _(np):
    vals = np.random.normal(-2,3,10000)
    return (vals,)


@app.cell
def _(Stats, vals):
    s = Stats()

    for v in vals:
        s.update(v)

    print(f"mean: {s.mean():3.3f}  var: {s.var():3.3f}  std: {s.std():3.3f}")
    return


if __name__ == "__main__":
    app.run()
