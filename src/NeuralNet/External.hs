module NeuralNet.External (loadExampleSet, createStdGenWeightsStream) where

import NeuralNet.Example
import NeuralNet.Net
import Text.CSV
import System.FilePath.Posix
import Data.Char
import System.Random


loadExampleSet :: FilePath -> IO ExampleSet
loadExampleSet p =
  case map toLower (takeExtension p) of
    ".csv" -> loadCsvExampleSet p
    _      -> error ("Don't know how to load example set from " ++ show p)

loadCsvExampleSet :: FilePath -> IO ExampleSet
loadCsvExampleSet p = do
  result <- parseCSVFromFile p
  case result of
    Right c -> return (createExampleSet (csvToExamples c))
    _       -> error ("Error reading" ++ p)

csvToExamples :: CSV -> [Example]
csvToExamples records = map (\r -> (init r, last r)) doubleRecords
  where doubleRecords = map (map read) records

createStdGenWeightsStream :: IO WeightsStream
createStdGenWeightsStream = do
  stdGen <- getStdGen
  return (randomRs (0, 0.1) stdGen)
