module NeuralNet.Example (
  Example,
  ExampleSet (),
  createExampleSet,
  exampleSetX,
  exampleSetY,
  exampleSetM,
  exampleSetN,
  splitExampleSet
) where

import Numeric.LinearAlgebra
import Data.List (nub)


type Example = ([Double], Double)

data ExampleSet = ExampleSet [Example] (Matrix Double) (Matrix Double)
  deriving (Show, Eq)

createExampleSet :: [Example] -> ExampleSet
createExampleSet e
  | numUniqueExampleSizes > 1  = error ("All examples must have same size (given " ++ show uniqueExampleSizes ++ ")")
  | numUniqueExampleSizes == 0 = error "No examples provided"
  | n == 0                     = error "Empty example provided"
  | otherwise                  = ExampleSet e x y
    where
      exampleSizes = map (length . fst) e
      uniqueExampleSizes = nub exampleSizes
      numUniqueExampleSizes = length uniqueExampleSizes
      n = head exampleSizes
      x = tr (fromLists (map fst e))
      y = fromLists [map snd e]

exampleSetX :: ExampleSet -> Matrix Double
exampleSetX (ExampleSet _ x _) = x

exampleSetY :: ExampleSet -> Matrix Double
exampleSetY (ExampleSet _ _ y) = y

exampleSetN :: ExampleSet -> Int
exampleSetN = rows . exampleSetX

exampleSetM :: ExampleSet -> Int
exampleSetM (ExampleSet e _ _) = length e

splitExampleSet :: Double -> ExampleSet -> (ExampleSet, ExampleSet)
splitExampleSet ratio (ExampleSet e _ _) = (createExampleSet leftExamples, createExampleSet rightExamples)
  where
    (leftExamples, rightExamples) = splitAt (round (ratio * fromIntegral (length e))) e

