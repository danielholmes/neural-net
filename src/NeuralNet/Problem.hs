module NeuralNet.Problem (Problem (), createProblem, runProblem) where

import NeuralNet.Example
import NeuralNet.Net


type NumIterations = Int

type LearningRate = Double

data Problem = Problem NeuralNetDefinition ExampleSet LearningRate NumIterations


type IterationNum = Int

type Cost = Double

type Accuracy = Double

data RunStep = RunStep IterationNum Cost Accuracy

createProblem :: NeuralNetDefinition -> ExampleSet -> LearningRate -> NumIterations -> Problem
createProblem def examples learningRate numIterations
  | not (isExampleSetCompatibleWithNNDef examples def) = error "Examples not compatible with nn"
  | otherwise                                          = Problem def examples learningRate numIterations

runProblem :: Problem -> (Double -> Double) -> (NeuralNet, [RunStep])
runProblem _ _ = (error "TODO", [RunStep 1 1 1])
