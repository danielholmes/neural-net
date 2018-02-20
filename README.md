# Neural Net

[![Build Status](https://travis-ci.org/danielholmes/neural-net.svg?branch=master)](https://travis-ci.org/danielholmes/neural-net)

A simple Neural Net implementation


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

Cat image example:

```bash
stack exec image-eg
```
