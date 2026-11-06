"""Position of a leprosy case on the Ridley-Jopling spectrum.

A tree ensemble trained on the cohort's Ridley-Jopling labels measures how
similar a record is to each of 26 specialist reference cases; the position is
the similarity-weighted mean of the reference cases' anchors, rescaled to the
range the fitted model can reach.
"""
