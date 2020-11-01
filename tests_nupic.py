#! /usr/bin/python
from nupic.algorithms.temporal_memory import TemporalMemory
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakSequenceMemory as TM

def main():

    tm0 = TM()

    tm = TemporalMemory(
        # Must be the same dimensions as the SP
        columnDimensions=(32768,),
        # How many cells in each mini-column.
        cellsPerColumn=4,
        # A segment is active if it has >= activationThreshold connected synapses
        # that are active due to infActiveState
        activationThreshold=1,#1,4(melhor),
        #initialPermanence=0.4,
        connectedPermanence=0,
        # Minimum number of active synapses for a segment to be considered during
        # search for the best-matching segments.
        minThreshold=1, #1
        # The max number of synapses added to a segment during learning
        maxNewSynapseCount=1, #6
        #permanenceIncrement=0.1,
        #permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0005,#0.0001,#0.0005,
        maxSegmentsPerCell=1, #8 16(colou)
        maxSynapsesPerSegment=1, #8 16(colou)
        seed=42
    )


if __name__ == "__main__":
    main()