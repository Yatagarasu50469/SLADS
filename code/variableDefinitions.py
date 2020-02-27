#! /usr/bin/env python3

class TrainingInfo:
    def initialize(self,ReconMethod,FeatReconMethod,p,NumNbrs,FilterType,FilterC,FeatDistCutoff,MaxWindowForTraining,*args):
        self.ReconMethod = ReconMethod
        self.FeatReconMethod = FeatReconMethod
        self.p = p
        self.NumNbrs = NumNbrs        
        self.FilterType = FilterType
        self.FilterC = FilterC
        self.FeatDistCutoff = FeatDistCutoff
        self.MaxWindowForTraining=MaxWindowForTraining
        if args:
            self.PAP_Iter=args[0]
            self.PAP_Beta=args[1]
            self.PAP_InitType=args[2]
            self.PAP_ScaleMax=args[3]
class InitialMask:
    def initialize(self,RowSz,ColSz,MaskType,MaskNumber,Percentage):
        self.RowSz = RowSz
        self.ColSz = ColSz
        self.MaskType = MaskType
        self.MaskNumber = MaskNumber
        self.Percentage = Percentage

class StopCondParams:
    def initialize(self,Beta,Threshold,JforGradient,MinPercentage,MaxPercentage):
        self.Beta = Beta
        self.Threshold = Threshold
        self.JforGradient = JforGradient
        self.MinPercentage = MinPercentage
        self.MaxPercentage = MaxPercentage

class UpdateERDParams:
    def initialize(self,Do,MinRadius,MaxRadius,IncreaseRadiusBy):
        self.Do = Do
        self.MinRadius = MinRadius
        self.MaxRadius = MaxRadius
        self.IncreaseRadiusBy = IncreaseRadiusBy


class BatchSamplingParams:
    def initialize(self,Do,NumSamplesPerIter):
        self.Do = Do
        self.NumSamplesPerIter = NumSamplesPerIter
        
    
