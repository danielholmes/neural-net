module NeuralNet.Problem (
  Problem (),
  RunStep (),
  createProblem,
  runProblem,
  runStepIteration,
  runStepAccuracy,
  runStepCost,
  problemTestSet
) where

import NeuralNet.Example
import NeuralNet.Net
import NeuralNet.Layer
import NeuralNet.Cost
import NeuralNet.Train
import Data.Matrix


type NumIterations = Int
type LearningRate = Double
data Problem = Problem NeuralNetDefinition ExampleSet ExampleSet LearningRate NumIterations
  deriving (Show, Eq)

type IterationNum = Int
type Cost = Double
type Accuracy = Double
data RunStep = RunStep IterationNum Cost Accuracy
  deriving (Show, Eq)

runStepIteration :: RunStep -> IterationNum
runStepIteration (RunStep i _ _) = i

runStepCost :: RunStep -> Cost
runStepCost (RunStep _ c _) = c

runStepAccuracy :: RunStep -> Accuracy
runStepAccuracy (RunStep _ _ a) = a

problemNNDef :: Problem -> NeuralNetDefinition
problemNNDef (Problem d _ _ _ _) = d

problemTrainSet :: Problem -> ExampleSet
problemTrainSet (Problem _ t _ _ _) = t

problemTestSet :: Problem -> ExampleSet
problemTestSet (Problem _ _ t _ _) = t

problemLearningRate :: Problem -> LearningRate
problemLearningRate (Problem _ _ _ l _) = l

problemNumIterations :: Problem -> NumIterations
problemNumIterations (Problem _ _ _ _ i) = i

createProblem :: NeuralNetDefinition -> ExampleSet -> ExampleSet -> LearningRate -> NumIterations -> Problem
createProblem def trainSet testSet learningRate numIterations
  | not (isExampleSetCompatibleWithNNDef trainSet def) = error "trainSet not compatible with nn"
  | not (isExampleSetCompatibleWithNNDef testSet def)  = error "testSet not compatible with nn"
  | numIterations <= 0                                 = error "Must provide positive numIterations"
  | learningRate <= 0                                  = error "Must provide positive learningRate"
  | otherwise                                          = Problem def trainSet testSet learningRate numIterations

runProblem :: WeightsStream -> Problem -> (Double -> Double -> Bool) -> (NeuralNet, [RunStep], Double)
runProblem g p accuracyCheck = (resultNN, tail allSteps, testAccuracy)
  where
    startNN = initNN g (problemNNDef p)
    startStep = RunStep 0 1 0
    iterations = [1..(problemNumIterations p)]
    allNNAndSteps = reverse (foldl (\steps@((nn,_):_) i -> runProblemStep p i nn accuracyCheck : steps) [(startNN, startStep)] iterations)
    allSteps = map snd allNNAndSteps
    resultNN = fst (last allNNAndSteps)

    testSet = problemTestSet p
    testA = forwardPropA (last (nnForwardSet resultNN testSet))
    testAccuracy = calcAccuracy accuracyCheck testA (exampleSetY testSet)

runProblemStep :: Problem -> IterationNum -> NeuralNet -> (Double -> Double -> Bool) -> (NeuralNet, RunStep)
runProblemStep p i nn accuracyCheck = (newNN, RunStep i cost accuracy)
  where
    trainSet = problemTrainSet p
    forwardSteps = nnForwardSet nn trainSet
    al = forwardPropA (last forwardSteps)
    y = exampleSetY trainSet
    grads = nnBackward nn forwardSteps trainSet
    newNN = updateNNParams nn grads (problemLearningRate p)
    cost = computeCost al y
    accuracy = calcAccuracy accuracyCheck al y

calcAccuracy :: (Double -> Double -> Bool) -> Matrix Double -> Matrix Double -> Double
calcAccuracy accuracyCheck yHat y = fromIntegral numCorrect / fromIntegral (ncols y)
  where
    numCorrect = length (filter (uncurry accuracyCheck) (zip (toList yHat) (toList y)))
