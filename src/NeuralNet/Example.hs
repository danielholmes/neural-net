module NeuralNet.Example (
  Example,
  ExampleSet (),
  createExampleSet,
  exampleSetX,
  exampleSetY,
  exampleSetN
) where

import Data.Matrix
import Data.List (nub)


type Example = ([Double], Double)

data ExampleSet = ExampleSet [Example] (Matrix Double) (Matrix Double)

createExampleSet :: [Example] -> ExampleSet
createExampleSet e
  | numUniqueExampleSizes > 1  = error "All examples must have same size"
  | numUniqueExampleSizes == 0 = error "No examples provided"
  | n == 0                     = error "Empty example provided"
  | otherwise                  = ExampleSet e x y
    where
      exampleSizes = map (length . fst) e
      numUniqueExampleSizes = length (nub exampleSizes)
      m = length e
      n = head exampleSizes
      x = matrix n m (\(r, c) -> (fst (e!!(c - 1)))!!(r - 1))
      y = fromList m 1 (map snd e)

exampleSetX :: ExampleSet -> Matrix Double
exampleSetX (ExampleSet _ x _) = x

exampleSetY :: ExampleSet -> Matrix Double
exampleSetY (ExampleSet _ _ y) = y

exampleSetN :: ExampleSet -> Int
exampleSetN = nrows . exampleSetX
