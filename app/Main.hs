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
import Data.List.Split


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
  let (nn, steps, testAccuracy) = runProblem weights problem (\yh y -> ((round yh) :: Int) == ((round y) :: Int))
  putStrLn (intercalate "\n" (map formatStepLine (every 10 (drop 9 steps))))
  putStrLn "Done in 0ms"
  print nn
  putStrLn ("Train Accuracy: " ++ formatPercent (runStepAccuracy (last steps)))
  putStrLn ("Test Accuracy: " ++ formatPercent testAccuracy)

formatStepLine :: RunStep -> String
formatStepLine s = show (runStepIteration s) ++ ") " ++ show (runStepCost s) ++ " " ++ formatPercent (runStepAccuracy s)

formatPercent :: Double -> String
formatPercent r = show (round (100.0 * r) :: Int) ++ "%"

createWeightsStream :: RunOptions -> IO WeightsStream
createWeightsStream options = case constInitWeights options of
  True -> return (repeat 0)
  False -> do
            stdGen <- getStdGen
            return (randomRs (0, 0.1) stdGen)
            --return (randomRs (0, 0.1) (read "1 0" :: StdGen)) TODO: Allow passing an optional seed for weights generator

loadProblem :: RunOptions -> IO Problem
loadProblem (RunOptions {trainPath = train, testPath = test, numIterations = n, learningRate = a}) = do
  trainSet <- loadExampleSet train
  testSet <- loadExampleSet test
  let nnDef = createLogRegDefinition (exampleSetN testSet) Sigmoid
  return (createProblem nnDef trainSet testSet a n)

every :: Int -> [a] -> [a]
every n = map head . chunksOf n
