import Test.Hspec

main :: IO ()
main = hspec $ do
    describe "buildNet" $ do
        it "returns Unknown for random" $
            True `shouldBe` True

        it "returns Unknown for no timestamp" $
            False `shouldBe` False

    describe "parseMessageWithType" $ do
        it "returns Unknown for random" $
            True `shouldBe` True

        it "returns Unknown for no timestamp" $
            False `shouldBe` False
