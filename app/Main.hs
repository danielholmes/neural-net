{-# LANGUAGE DeriveDataTypeable #-}
{-# OPTIONS_GHC -fno-cse #-}
module Main where

import NeuralNet.Problem
import NeuralNet.Net
import NeuralNet.Activation
import NeuralNet.Example
import NeuralNet.External
import System.IO ()
import System.Random
import Data.List
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
  gen <- createWeightInit options
  let (nn, steps) = runProblem gen problem (\yh y -> ((round yh) :: Int) == ((round y) :: Int))
  putStrLn (intercalate "\n" (map formatStepLine steps))
  print nn

formatStepLine :: RunStep -> String
formatStepLine s = show (runStepIteration s) ++ ") " ++ show (runStepCost s) ++ " " ++ show accuracy ++ "%"
  where accuracy = (round (100.0 * runStepAccuracy s)) :: Int

createWeightInit :: RunOptions -> IO WeightInitialiser
createWeightInit options = case constInitWeights options of
  True -> return (Const 0)
  False -> do
            stdGen <- getStdGen
            return (Random stdGen)

loadProblem :: RunOptions -> IO Problem
loadProblem (RunOptions {trainPath = train, testPath = test, numIterations = n, learningRate = a}) = do
  trainSet <- loadExampleSet train
  testSet <- loadExampleSet test
  let nnDef = createLogRegDefinition (exampleSetN testSet) Sigmoid
  return (createProblem nnDef trainSet testSet a n)
