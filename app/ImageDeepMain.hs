{-# LANGUAGE DeriveDataTypeable #-}
{-# OPTIONS_GHC -fno-cse #-}
module ImageDeepMain (main) where

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
                             ,weightsSeed :: Maybe String
                             ,numIterations :: Int
                             ,learningRate :: Double}
  deriving (Eq, Show, Data, Typeable)

optionsDef :: RunOptions
optionsDef = RunOptions
  {trainYesDir = def &= typ "TRAINYESDIR" &= argPos 0
  ,trainNoDir = def &= typ "TRAINNODIR" &= argPos 1
  ,testYesDir = def &= typ "TESTYESDIR" &= argPos 2
  ,testNoDir = def &= typ "TESTNODIR" &= argPos 3
  ,weightsSeed = Nothing &= name "w" &=name "weights-seed" &= help "A filepath to a CSV of weights, or an Int to seed random weights"
  ,numIterations = 250 &= name "i" &= name "num-iterations" &= help "Number of training iterations"
  ,learningRate = 0.005 &= name "l" &= name "learning-rate" &= help "Learning rate (alpha)"} &=
  help "Simple Neural Net trainer" &=
  summary "NeuralNet v0.0.0, (C) Daniel Holmes"

main :: IO ()
main = do
  options <- cmdArgs optionsDef
  problem <- loadProblem options
  weights <- createWeightsStream options
  uiRunProblem problem weights

createWeightsStream :: RunOptions -> IO [Double]
createWeightsStream (RunOptions { weightsSeed = maybeSeed }) = case maybeSeed of
  Nothing -> createStdGenWeightsStream
  Just seed -> createCsvWeightsStream seed -- TODO: Configure no weights scaling in initNN
  -- Just seed -> createSeededGenWeightsStream seed
  -- TODO: If file exists, use as file. If an int, use as int. Else throw error

loadProblem :: RunOptions -> IO Problem
loadProblem options = do
  trainSet <- loadImageSet (trainYesDir options) (trainNoDir options)
  testSet <- loadImageSet (testYesDir options) (testNoDir options)
  let nnDef = (exampleSetN trainSet, [LayerDefinition ReLU 20, LayerDefinition ReLU 7, LayerDefinition ReLU 5, LayerDefinition Sigmoid 1])
  return (createProblem nnDef trainSet testSet (learningRate options) (numIterations options))
