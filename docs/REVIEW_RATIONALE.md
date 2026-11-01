# Why a review page, and a 60 to 90 second demo outline

## Research rationale

GoTrain trains a model without a human in the loop. The next person to touch
that model is an analyst who has to decide which outputs to look at. The review
page answers one question for them: "where might this model be wrong, and which
predictions should I inspect first?"

The direction is inspired by published priorities of the sponsor and the local
research community; none of these sources asked for this feature or endorses it:

- RASID's GoPilot emphasizes transparent, reproducible geospatial workflows that
  return actionable outputs (https://rasid.ai/about), and its BananaSight case
  study describes directing human inspection toward suspicious areas
  (https://rasid.ai/case-studies/bananasight-tr4-lebanon).
- The GEOAI group at the Lebanese National Center for Remote Sensing (CNRS)
  (https://geogroup.ai/) describes a research direction on explainable agents and a
  collaboration with RASID in its August 12, 2026 newsletter
  (https://geoai.substack.com/p/geoai-august-2026-newsletter), and its July 2026
  entropy-centric segmentation work emphasizes explainability and evaluation
  (https://geogroup.ai/publication/2026etecom_entropycentric/). GEOAI has also
  published Lebanese rooftop and solar-potential mapping
  (https://geogroup.ai/publication/2022solarmaplebanon/).

What GoTrain implements is deliberately simple: per-pixel predictive entropy of
the selected model's own softmax output, aggregated per image as the mean of the
most uncertain 10 percent of pixels, used only to order a review queue. It is an
uncertainty heuristic. It does not explain the model's reasoning, it is not a
reproduction of the GEOAI entropy-centric work, it is not calibrated correctness,
and a confidently wrong prediction will not be flagged. The retrospective benchmark
(share of misclassified pixels captured within a 10, 20 or 30 percent review budget
versus 200 random orders) is exploratory evidence on the German pv4ger validation
set only. It says nothing about Lebanese rooftops, rooftop suitability or energy
yield, and it does not measure reviewer time saved.

## Demo outline (60 to 90 seconds)

1. (10 s) "GoTrain trained this solar-panel segmentation model autonomously on
   SageMaker; validation mIoU 0.923. The analyst who receives it needs to know what
   to check first."
2. (15 s) Open `review.html` from the file browser. "One file, works offline. The
   queue is ordered by how unsure the model's own output was; nothing here used the
   correct masks to rank."
3. (20 s) Click the first item: original image, predicted solar regions in blue,
   uncertainty in yellow to red. Toggle the correct mask (validation demo) to show
   where the prediction and truth disagree. Mark "Needs correction", type a note.
4. (15 s) Press `j` a few times, mark two more, export the notes JSON. "The export
   carries the model version and checksum, so notes can never be mixed up with
   another model."
5. (15 s) Expand "Technical evidence": "At a 10 percent review budget the priority
   order contains 37 percent of all misclassified pixels; random orders average 10
   percent, with a 95th percentile of 14. Same picture at 20 and 30 percent. This is
   a retrospective check on the validation set, not a guarantee."
6. (10 s) "Everything was built from the stored model version with checksum and
   reference-prediction checks; the test split was never touched."

Numbers in step 5 are from the CPU build in REVIEW_VERIFIED.md; replace them with
the SageMaker build's numbers if they differ.
