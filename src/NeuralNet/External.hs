module NeuralNet.External (
  loadExampleSet,
  createStdGenWeightsStream,
  loadImageSet
) where

import NeuralNet.Example
import NeuralNet.Net
import Text.CSV
import System.FilePath.Posix
import Data.Char
import System.Random
import System.Random.Shuffle
import qualified Data.Vector.Storable as Vector
import System.Directory
import Codec.Picture
import System.IO ()


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

loadImageSet :: FilePath -> FilePath -> IO ExampleSet
loadImageSet yesDir noDir = do
  yes <- loadImageExamples yesDir 1
  no <- loadImageExamples noDir 0
  gen <- getStdGen
  let examples = yes ++ no
  let orderedExamples = shuffle' (yes ++ no) (length examples) gen
  return (createExampleSet orderedExamples)

loadImageExamples :: FilePath -> Double -> IO [Example]
loadImageExamples p y = do
  names <- listDirectory p
  mapM (\n -> loadImageExample y (p ++ "/" ++ n)) (filter (\n -> head n /= '.') names)

loadImageExample :: Double -> FilePath -> IO Example
loadImageExample y f = do
  imageResult <- readImage f
  case imageResult of
    Left message   -> error ("Error on " ++ f ++ ": " ++ message)
    Right rawImage -> do
      let image = convertRGB8 rawImage
      let pixels = map (\p -> fromIntegral p / 255.0) (Vector.toList (imageData image))
      return (pixels, y)
