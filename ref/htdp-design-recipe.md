# The HtDP Function Design Recipe

A practical reference for designing programs systematically.  
Based on *How to Design Programs* (2nd edition).

Six steps. Do them in order. Each step produces something the next step uses.  
If you get stuck, go back one step.

---

## The Six Steps

The design recipe is a checklist for writing a function. It keeps you from staring at a blank screen and inventing code. You decide what data you have, what the function promises, what it should do on concrete examples, and only then write the body.

| Step | Name | What you produce |
|------|------|------------------|
| 1 | Data definitions | What information exists and how it is represented |
| 2 | Signature, purpose, stub | The contract of the function + a runnable placeholder |
| 3 | Functional examples | Concrete input → expected output pairs |
| 4 | Template | A skeleton driven by the shape of the data |
| 5 | Function definition | The real body, filling the template |
| 6 | Tests | The examples from Step 3, run against the finished function |

**Golden rule.** Do not jump to code. The missing piece is almost always in the data definition or the examples.

---

## Step 1 — Data Definitions

Identify the information the problem talks about and decide how that information will be represented in the programming language. Write a data definition for each class of data, plus a short interpretation of what the values mean.

### Questions to answer

- What real-world things, quantities, or concepts matter?
- Which of those does the program actually need?
- How will each piece be represented (number, string, structure, list, …)?
- What are the variants or cases? (atomic, enumeration, compound, one-of, self-referential)

### Shapes of data

The shape you choose here decides the template in Step 4. Almost every program is a combination of these.

| Shape | Meaning | Example |
|-------|---------|---------|
| Atomic | One indivisible value. No parts. | Number, String, Boolean |
| Enumeration | A finite list of distinct known values. | `"red"` / `"yellow"` / `"green"` |
| Compound (structure) | Several pieces that belong together. | A point with x and y; a pair of flags |
| One-of / itemization | The value is one of several different kinds. | A plot or nothing; a cat or a dog or `"unknown"` |
| Self-referential | A definition that mentions itself. For data of unknown size. | A list of numbers; a family tree |

### Example — temperature message

Problem: given a room temperature, say whether it is cold, comfortable, or hot.

```racket
; A Temperature is a Number.
; interpretation: degrees Celsius

; A Comfort-Status is one of:
; – "cold"
; – "comfortable"
; – "hot"
; interpretation: how the temperature feels
```

Examples of the data itself (not of the function yet):

```racket
; Temperature examples
20
-5
37.5

; Comfort-Status examples
"cold"
"comfortable"
"hot"
```

### Example — plots controlled by flags

Problem: generate a scatter plot and/or a histogram from a CSV, only when the user asks for them.

```python
# A Dataset is a pandas.DataFrame.
# interpretation: each row is one observation; columns are variables.
# Required columns:
#   column1 : Number
#   column2 : Number

from dataclasses import dataclass

@dataclass
class ControlFlags:
    show_scatter: bool
    show_histogram: bool
# interpretation: values from UI widgets that decide which plots to generate

# A ScatterPlot is a plotly Figure (or None if not requested)
# A Histogram is a plotly Figure (or None if not requested)
```

**Note:** if the program is interactive, the control values (booleans, selections) are also information. Put them in Step 1.

---

## Step 2 — Signature, Purpose, Stub

State what data the function consumes and produces. Write one sentence that says what the function computes. Define a stub that matches the signature and returns a dummy value of the correct type.

### Signature

The types of the inputs and the type of the output. In HtDP it is a comment with an arrow. In Python it is a type hint.

```racket
; generate-plots : Dataset ControlFlags -> (ScatterPlot or #false) (Histogram or #false)
```

```python
# Python
def generate_plots(
    df: Dataset,
    flags: ControlFlags,
) -> tuple[ScatterPlot | None, Histogram | None]:
```

The names on the left and right must come from Step 1 or from built-in types. `Dataset` is not a built-in type; it is a name you invented in Step 1.

### Purpose statement

One short sentence. Say *what* is computed, not *how*.

- Good: “Generates a scatter plot and/or a histogram from the dataset according to the control flags. Returns None for any plot that should not be shown.”
- Bad: “Uses plotly.express and two if statements to draw charts.”

### Header / stub

A function with the right name and parameters, plus a dummy return of the correct type so the program still runs. The stub is not the real body; it only holds the place so you can write examples next.

```python
def generate_plots(df: Dataset, flags: ControlFlags) -> tuple[ScatterPlot | None, Histogram | None]:
    """Generates a scatter plot and/or a histogram from the given dataset,
    according to the control flags. Returns None for any plot that should not be shown."""
    return (None, None)   # stub
```

---

## Step 3 — Functional Examples

Work through concrete input → expected output pairs that illustrate the purpose. Write them *before* you implement the function. Cover typical cases, every variant of mixed data, and boundary cases.

This is not implementation. Do not write the real body here. This is also not the final test suite; you turn these sketches into executable tests in Step 6.

```python
df1 = ...  # a small Dataset

# both plots requested
generate_plots(df1, ControlFlags(True, True))
# expected: (a ScatterPlot of column1 vs column2, a Histogram of column1)

# only scatter
generate_plots(df1, ControlFlags(True, False))
# expected: (a ScatterPlot, None)

# only histogram
generate_plots(df1, ControlFlags(False, True))
# expected: (None, a Histogram)

# neither
generate_plots(df1, ControlFlags(False, False))
# expected: (None, None)
```

If the function returns a tuple `(scatter, histo)`, index 0 is the first element and index 1 is the second.

---

## Step 4 — Template

Translate the data definitions into an outline of the function. The template is a skeleton driven by the *shape of the data*, not by an algorithm you already have in mind.

### Rules of thumb

- Atomic data → use the parameter as-is.
- Compound data → write a selector for every field.
- One-of / enumeration → one conditional clause per variant.
- Self-referential data → process the first piece and call the same function on the rest.

```python
def generate_plots(df: Dataset, flags: ControlFlags) -> tuple[ScatterPlot | None, Histogram | None]:
    # template based on ControlFlags
    # flags.show_scatter   : Boolean
    # flags.show_histogram : Boolean

    if flags.show_scatter:
        scatter = ...          # produce a ScatterPlot from df
    else:
        scatter = None

    if flags.show_histogram:
        histo = ...            # produce a Histogram from df
    else:
        histo = None

    return (scatter, histo)
```

The “...” means “not filled in yet.” That is correct for Step 4.

---

## Step 5 — Function Definition

Fill the gaps in the template using the purpose statement, the data definitions, and the examples. Replace each “...” with an expression that produces the right type.

```python
def generate_plots(df: Dataset, flags: ControlFlags) -> tuple[ScatterPlot | None, Histogram | None]:
    """Generates a scatter plot and/or a histogram from the given dataset,
    according to the control flags. Returns None for any plot that should not be shown."""

    if flags.show_scatter:
        scatter = px.scatter(df, x="column1", y="column2")
    else:
        scatter = None

    if flags.show_histogram:
        histo = px.histogram(df, x="column1")
    else:
        histo = None

    return (scatter, histo)
```

If a hole is still hard to fill, you probably need a helper function. Design the helper with the same six steps. Do not invent extra logic that the data definition does not justify.

---

## Step 6 — Tests

Turn the examples from Step 3 into executable checks and run them against the finished function.

Step 3 is written before the body and may be comments. Step 6 is run after the body exists and must actually execute.

```python
df = pd.DataFrame({
    "column1": [1.2, 2.5, 0.8],
    "column2": [3.4, 5.1, 2.9]
})

scatter, histo = generate_plots(df, ControlFlags(True, True))
assert scatter is not None and histo is not None

scatter, histo = generate_plots(df, ControlFlags(True, False))
assert scatter is not None and histo is None

scatter, histo = generate_plots(df, ControlFlags(False, True))
assert scatter is None and histo is not None

scatter, histo = generate_plots(df, ControlFlags(False, False))
assert scatter is None and histo is None
```

If a test fails, do not immediately rewrite the whole function. Ask:

- Is the example wrong?
- Is the data definition incomplete?
- Did the template miss a case?
- Is only the filled-in expression wrong?

---

## Working Checklist

Copy this onto a new problem and tick the boxes as you go.

- [ ] 1. Information listed
- [ ] 1. Representation chosen
- [ ] 1. Data definitions + interpretation written
- [ ] 1. Examples of the data itself
- [ ] 2. Signature written
- [ ] 2. Purpose statement (what, not how)
- [ ] 2. Stub returns a dummy value of the right type
- [ ] 3. Examples cover every variant and boundary
- [ ] 4. Template follows the data shape (selectors + conditionals)
- [ ] 5. Holes filled; helpers get their own recipe
- [ ] 6. Examples are executable and have been run

### Typical mistakes

- Writing library calls or the real body in Step 1 or Step 3.
- Treating Step 3 examples as the finished implementation.
- Forgetting UI or control flags in the data definitions.
- Using invented names (`Dataset`) in annotations without defining them.
- Testing only the happy path.
- Jumping to code because you already know the library.

After the tests pass, review and refactor if needed. Collapse duplication, extract helpers, rename for clarity. Do not refactor *instead of* the first six steps.

---

*Source: How to Design Programs, 2nd edition (Felleisen et al.). The six steps are the function design recipe. Python notes are practical adaptations, not part of the original book.*
