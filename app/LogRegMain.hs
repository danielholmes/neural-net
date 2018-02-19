{-# LANGUAGE DeriveDataTypeable #-}
{-# OPTIONS_GHC -fno-cse #-}
module LogRegMain (main) where

import NeuralNet.Problem
import NeuralNet.Net
import NeuralNet.Activation
import NeuralNet.Example
import NeuralNet.External
import NeuralNet.UI
import System.IO ()
import System.Console.CmdArgs


data RunOptions = RunOptions {trainPath :: FilePath
                             ,testPath :: FilePath
                             ,constInitWeights :: Bool
                             ,numIterations :: Int
                             ,learningRate :: Double}
  deriving (Eq, Show, Data, Typeable)

optionsDef :: RunOptions
optionsDef = RunOptions
  {trainPath = def &= typ "TRAINFILE" &= argPos 0
  ,testPath = def &= typ "TESTFILE" &= argPos 1
  ,constInitWeights = def &= name "c" &= name "const-init-weights" &= help "Use 0 for initial weights"
  ,numIterations = 1000 &= name "i" &= name "num-iterations" &= help "Number of training iterations"
  ,learningRate = 0.005 &= name "l" &= name "learning-rate" &= help "Learning rate (alpha)"} &=
  help "Simple Neural Net trainer" &=
  summary "NeuralNet v0.0.0, (C) Daniel Holmes"

main :: IO ()
main = do
  options <- cmdArgs optionsDef
  problem <- loadProblem options
  weights <- createWeightsStream options
  uiRunProblem problem weights

createWeightsStream :: RunOptions -> IO WeightsStream
createWeightsStream options = case constInitWeights options of
  True  -> return (repeat 0)
  False -> createStdGenWeightsStream
  --return (randomRs (0, 0.1) (read "1 0" :: StdGen)) TODO: Allow passing an optional seed for weights generator

loadProblem :: RunOptions -> IO Problem
loadProblem (RunOptions {trainPath = train, testPath = test, numIterations = n, learningRate = a}) = do
  trainSet <- loadExampleSet train
  testSet <- loadExampleSet test
  let nnDef = createLogRegDefinition (exampleSetN testSet) Sigmoid
  return (createProblem nnDef trainSet testSet a n)
