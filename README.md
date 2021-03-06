# Neural Net

[![Build Status](https://travis-ci.org/danielholmes/neural-net.svg?branch=master)](https://travis-ci.org/danielholmes/neural-net)

A simple binary classification Neural Net implementation


## Dependencies

 - [Stack](https://www.haskellstack.org/)


## Setup
```bash
stack init
```


## Build
```bash
stack build --pedantic
```

Continuous:

```bash
stack build --pedantic --file-watch
```

Only one executable:

```bash
stack build neural-net:exe:image-eg --pedantic --file-watch
```


## Running Tests
```bash
stack test --pedantic
```

With watching of files for changes
```bash
stack test --file-watch --pedantic
```

With error stack traces
```bash
stack test --pedantic --file-watch --profile
```

With more tracing
```bash
stack test --pedantic --file-watch --trace .
```


## Running

```bash
stack exec neural-net-exe -- --help
```

Simple Logistic Regression example:

```bash
stack exec logreg -- -c -l 0.005 -i 1000 examples/simple-csv/train.csv examples/simple-csv/test.csv
```

Binary image classification Log Reg example (requires copying in some files to `/examples`):

```bash
stack exec image-logreg -- -l 0.005 -i 1000 \
    examples/images/nn-course-train/1 \
    examples/images/nn-course-train/0 \
    examples/images/nn-course-test/1 \
    examples/images/nn-course-test/0
```

Binary image classification Deep NN example (requires copying in some files to `/examples`):

```bash
stack exec image-deep -- -l 0.0075 -i 2500 \
    examples/images/nn-course-train/1 \
    examples/images/nn-course-train/0 \
    examples/images/nn-course-test/1 \
    examples/images/nn-course-test/0
```


## Running with Time Profiling

```bash
stack build neural-net:exe:image-deep --pedantic --profile
stack exec image-deep -- -l 0.0075 -i 2500 \
    examples/images/nn-course-train/1 \
    examples/images/nn-course-train/0 \
    examples/images/nn-course-test/1 \
    examples/images/nn-course-test/0 \
    +RTS -p
```
