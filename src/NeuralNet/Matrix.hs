module NeuralNet.Matrix (
  mapMatrix,
  broadcastCols,
  sumRows,
  sumMatrix
) where

import Data.Matrix
import qualified Data.Vector as Vector

mapMatrix :: (a -> a) -> Matrix a -> Matrix a
mapMatrix f m = foldr (mapRow (\_ x -> f x)) m [1..(nrows m)]

broadcastCols :: Matrix a -> Int -> Matrix a
broadcastCols m n = foldl (\mx _ -> mx <|> m) m [1..(n - 1)]

sumRows :: Num a => Matrix a -> Matrix a
sumRows m = foldl (\n r -> setElem (sum (Vector.toList (getRow r m))) (r, 1) n) base [1..size]
  where
    size = nrows m
    base = zero size 1

sumMatrix :: Num a => Matrix a -> a
sumMatrix = sum . toList
