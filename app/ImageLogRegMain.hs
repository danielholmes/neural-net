{-# LANGUAGE DeriveDataTypeable #-}
{-# OPTIONS_GHC -fno-cse #-}
module ImageLogRegMain (main) where

import NeuralNet.Example
import NeuralNet.Activation
import NeuralNet.Net
import NeuralNet.Problem
import NeuralNet.External
import NeuralNet.UI
import System.IO ()
import System.Console.CmdArgs


data RunOptions = RunOptions {trainYesDir :: FilePath
                             ,trainNoDir :: FilePath
                             ,testYesDir :: FilePath
                             ,testNoDir :: FilePath
                             ,constInitWeights :: Bool
                             ,numIterations :: Int
                             ,learningRate :: Double}
  deriving (Eq, Show, Data, Typeable)

optionsDef :: RunOptions
optionsDef = RunOptions
  {trainYesDir = def &= typ "TRAINYESDIR" &= argPos 0
  ,trainNoDir = def &= typ "TRAINNODIR" &= argPos 1
  ,testYesDir = def &= typ "TESTYESDIR" &= argPos 2
  ,testNoDir = def &= typ "TESTNODIR" &= argPos 3
  ,constInitWeights = def &= name "c" &= name "const-init-weights" &= help "Use 0 for initial weights"
  ,numIterations = 250 &= name "i" &= name "num-iterations" &= help "Number of training iterations"
  ,learningRate = 0.005 &= name "l" &= name "learning-rate" &= help "Learning rate (alpha)"} &=
  help "Simple Neural Net trainer" &=
  summary "NeuralNet v0.0.0, (C) Daniel Holmes"

main :: IO ()
main = do
  options <- cmdArgs optionsDef
  problem <- loadProblem options
  --weights <- createStdGenWeightsStream
  let weights = repeat 0
  uiRunProblem problem weights

loadProblem :: RunOptions -> IO Problem
loadProblem options = do
  trainSet <- loadImageSet (trainYesDir options) (trainNoDir options)
  testSet <- loadImageSet (testYesDir options) (testNoDir options)
  let nnDef = createLogRegDefinition (exampleSetN trainSet) Sigmoid
  return (createProblem nnDef trainSet testSet (learningRate options) (numIterations options))
