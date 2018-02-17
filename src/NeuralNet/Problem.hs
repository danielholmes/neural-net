module NeuralNet.Problem (Problem (), createProblem, runProblem) where

import System.Random
import NeuralNet.Example
import NeuralNet.Net
import NeuralNet.Layer
import NeuralNet.Cost
import NeuralNet.Train


type NumIterations = Int

type LearningRate = Double

data Problem = Problem NeuralNetDefinition ExampleSet LearningRate NumIterations


type IterationNum = Int

type Cost = Double

data RunStep = RunStep IterationNum Cost


problemNNDef :: Problem -> NeuralNetDefinition
problemNNDef (Problem d _ _ _) = d

problemExampleSet :: Problem -> ExampleSet
problemExampleSet (Problem _ e _ _) = e

problemLearningRate :: Problem -> LearningRate
problemLearningRate (Problem _ _ l _) = l

problemNumIterations :: Problem -> NumIterations
problemNumIterations (Problem _ _ _ i) = i

createProblem :: NeuralNetDefinition -> ExampleSet -> LearningRate -> NumIterations -> Problem
createProblem def examples learningRate numIterations
  | not (isExampleSetCompatibleWithNNDef examples def) = error "Examples not compatible with nn"
  | numIterations <= 0                                 = error "Must provide positive numIterations"
  | learningRate <= 0                                  = error "Must provide positive learningRate"
  | otherwise                                          = Problem def examples learningRate numIterations

runProblem :: StdGen -> Problem -> (NeuralNet, [RunStep])
runProblem g p = (resultNN, allSteps)
  where
    startNN = initNN g (problemNNDef p)
    startStep = RunStep 0 1
    iterations = [1..(problemNumIterations p)]
    allNNAndSteps = reverse (foldl (\steps@((nn,_):_) i -> runProblemStep p i nn : steps) [(startNN, startStep)] iterations)
    allSteps = map snd allNNAndSteps
    resultNN = fst (last allNNAndSteps)

runProblemStep :: Problem -> IterationNum -> NeuralNet -> (NeuralNet, RunStep)
runProblemStep p i nn = (newNN, RunStep i cost)
  where
    exampleSet = problemExampleSet p
    forwardSteps = nnForwardSet nn exampleSet
    al = forwardPropA (last forwardSteps)
    cost = computeCost al (exampleSetY exampleSet)
    grads = nnBackward nn forwardSteps exampleSet
    newNN = updateNNParams nn grads (problemLearningRate p)
