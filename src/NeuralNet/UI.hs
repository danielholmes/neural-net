module NeuralNet.UI (
  uiRunProblem
) where

import NeuralNet.Problem
import NeuralNet.Net
import Data.List.Split
import Control.StopWatch
import System.Clock


uiRunProblem :: Problem -> WeightsStream -> IO ()
uiRunProblem problem weights = do
  let op = return (runProblem weights problem (\yh y -> (round yh :: Int) == (round y :: Int)))
  ((nn, steps, testAccuracy), time) <- stopWatch op

  let reportEvery = if length steps > 100 then 10 else 1
  mapM_ (putStrLn . formatStepLine) (every reportEvery (drop (reportEvery - 1) steps))
  putStrLn ("Done in " ++ show (toNanoSecs time `div` 1000) ++ "ms")
  if nnNumInputs nn > 3
    then putStrLn "NN too large to print"
    else print nn
  putStrLn ("Train Accuracy: " ++ formatPercent (runStepAccuracy (last steps)))
  putStrLn ("Test Accuracy: " ++ formatPercent testAccuracy)

formatStepLine :: RunStep -> String
formatStepLine s = show (runStepIteration s) ++ ") " ++ show (runStepCost s) ++ " " ++ formatPercent (runStepAccuracy s)

formatPercent :: Double -> String
formatPercent r = show (round (100.0 * r) :: Int) ++ "%"

every :: Int -> [a] -> [a]
every n = map head . chunksOf n
