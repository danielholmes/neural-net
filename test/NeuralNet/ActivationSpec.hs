module NeuralNet.ActivationSpec (activationSpec) where

import Test.Hspec
import Data.Matrix
import NeuralNet.Activation


activationSpec :: SpecWith ()
activationSpec =
  describe "NeuralNet.Activation" $ do
    describe "forward" $ do
      it "should correctly work with ReLU" $
        let
          source = setElem (-1) (1, 2) (identity 4)
          expected = setElem 0 (1, 2) (identity 4)
        in forward ReLU source `shouldBe` expected

      it "should correctly work with Sigmoid" $
        let
          source = zero 4 4
          expected = matrix 4 4 (const 0.5)
        in forward Sigmoid source `shouldBe` expected

      it "should correctly work with Tanh" $
        let
          source = zero 4 4
          expected = matrix 4 4 (const 0)
        in forward Tanh source `shouldBe` expected

    describe "backward" $ do
      it "should correctly work with ReLU eg 1" $
        let
          dA = fromLists [[-0.41675785, -0.05626683]]
          z = fromLists [[0.04153939, -1.11792545]]
          expected = fromLists [[-0.41675785, 0]]
        in backward ReLU dA z `shouldBe` expected

      it "should correctly work with ReLU eg 2" $
        let
          dA = fromLists [[0.12913162, -0.44014127], [-0.14175655, 0.48317296], [0.01663708, -0.05670698]]
          z = fromLists [[-0.7129932, 0.62524497], [-0.16051336, -0.76883635], [-0.23003072, 0.74505627]]
          expected = fromLists [[0, -0.44014127], [0, 0], [0, -0.05670698]]
        in backward ReLU dA z `shouldBe` expected

--      it "should correctly work with Tanh" $
--        let
--          source = diagonalList 3 0 [1..]
--          expected = fromLists [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
--        in backward Tanh source `shouldBe` expected

      it "should correctly work with Sigmoid eg 1" $
        let
          dA = fromLists [[-0.41675785, -0.05626683]]
          z = fromLists [[0.04153939, -1.11792545]]
          expected = fromLists [[-0.1041445301481718, -1.0447915348382566e-2]]
        in backward Sigmoid dA z `shouldBe` expected

      it "should correctly work with Sigmoid eg 2" $
        let
          dA = fromLists [[-0.5590876, 1.77465392]]
          z = fromLists [[0.64667545, -0.35627076]]
          expected = fromLists [[-0.1261203972684239, 0.4298776138385045]]
        in backward Sigmoid dA z `shouldBe` expected
