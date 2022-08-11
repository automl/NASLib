# Datasheet for NAS-Bench-Suite-Zero

We include a [Datasheet](https://arxiv.org/abs/1803.09010). 
Thanks for the Markdown template from [Christian Garbin's repository](https://github.com/fau-masters-collected-works-cgarbin/datasheet-for-dataset-template).

Jump to section:

- [Motivation](#motivation)
- [Dataset Composition](#dataset-composition)
- [Collection Process](#collection-process)
- [Data Preprocessing](#data-preprocessing)
- [Data Distribution](#data-distribution)
- [Dataset Maintenance](#dataset-maintenance)
- [Legal and Ethical Considerations](#legal-and-ethical-considerations)

## Motivation

### Why was the datasheet created? (e.g., was there a specific task in mind? was there a specific gap that needed to be filled?)

The goal of our work is to make it easier and faster for researchers to run generalizable, reproducible ZC proxy experiments, and to motivate further study on exploiting the complementary strengths of ZC proxies. 
By pre-computing ZC proxies across many benchmarks, users can run many trials of NAS experiments cheaply on a CPU, 
reducing their carbon footprint [[ref1](https://arxiv.org/abs/2104.10350), [ref2](https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/)].
Since prior research in NAS has notoriously high GPU consumption [[ref3](https://arxiv.org/abs/1611.01578), [ref4](https://arxiv.org/abs/1802.01548)], this reduction in CO2 emissions is worthwhile.

### Has the dataset been used already? If so, where are the results so others can compare (e.g., links to published papers)?

The dataset has only been used in this paper. See Sections 4 and 5 and Appendix D and E.

### What (other) tasks could the dataset be used for?

Since the dataset only contains values of ZC proxies on existing NAS benchmarks, we are not aware of any tasks this dataset can be used for, 
besides analyzing ZC proxies and speeding up ZC proxy-based NAS algorithms.

### Who funded the creation of the dataset? 

This dataset was created by researchers at the University of Freiburg, Abacus.AI, the University of Toronto, and the Bosch Center for Artificial Intelligence. 
Funding for the dataset computation itself is from the University of Freiburg.

### Any other comments?

None.

## Dataset Composition

### What are the instances?(that is, examples; e.g., documents, images, people, countries) Are there multiple types of instances? (e.g., movies, users, ratings; people, interactions between them; nodes, edges)

For each NAS benchmark, each instance is a tuple of an architecture hash, the name of a ZC proxy, and the value and runtime of the ZC proxy evaluated on that architecture.

### How many instances are there in total (of each type, if appropriate)?

See Table 2 of our paper for a full breakdown of the number of instances for each NAS benchmark.

### What data does each instance consist of ? “Raw” data (e.g., unprocessed text or images)? Features/attributes? Is there a label/target associated with instances? If the instances related to people, are subpopulations identified (e.g., by age, gender, etc.) and what is their distribution?

Each instance is a tuple of an architecture hash, the name of a ZC proxy, and the value and runtime of the ZC proxy evaluated on that architecture. 
These will most-often be used to speed up NAS experiments or run analysis on ZC proxies, in which case they are not used as features/labels.

### Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

There is no missing information from individual instances.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.

There are no relationships between individual instances.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).

NAS-Bench-201 and TransNAS-Bench-101-Micro and Macro contain all possible instances.
NAS-Bench-101, NAS-Bench-301, and the additional architectures evaluated on spherical-cifar, SVHN, and NinaPro are samples.
All samples are drawn uniformly at random from the respective search space.
This is ensured because the code used to draw architectures uniformly at random is from the respective original repositories that introduced the NAS benchmarks.

### Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.

The main usage of this dataset is to speed up NAS experiments, for which there are no data splits.
For experiments involving architecture prediction (such as the standalone predictor experiments in Section 5, 
we do not give recommended data splits but instead recommended running at least 100 trials, 
where each trial randomly samples train and (disjoint) test sets.

### Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.

There are no known errors, sources of noise, or redundancies.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

The dataset does rely on the code from the respective existing NAS benchmarks to reconstruct the architecture itself from the hash provided in our dataset. 
Furthermore, a user will often want access to the validation accuracies of the architectures in our dataset, which also comes from the existing NAS benchmarks.
Since these NAS benchmarks serve similar goals as our dataset (to accelerate and simplify research in NAS) and are hosted similarly to ours (on Google Drive and GitHub), 
we are confident that these benchmarks will exist and remain constant over time. 
In some cases, we have also created our own versions of the NAS benchmarks, so all of the data can be downloaded at one time.
Licenses and links are described in Table 4 of our paper.

### Any other comments?

None.


## Collection Process


### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?
 
The data was created with a software program (available at [https://github.com/automl/NASLib/tree/zerocost](https://github.com/automl/NASLib/tree/zerocost)).
The ZC proxy code were taken from their original repositories.
All ZC proxies from Table 1 were run on an Intel Xeon Gold 6242 CPU, using a batch size of 64, except for the case of TransNAS-Bench-101: due to the extreme memory usage of the Taskonomy tasks ($>30$GB memory), we used a batch size of 32. 


### How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.
 
As described, all data was created with a publicly available software program.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
 
As described earlier, the sampling was done uniformly at random.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
 
The data collection process (e.g., running the code) was done by the authors of this work.

### Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.
 
The total computation time for all 1.5M evaluations was 1100 CPU hours on Intel Xeon Gold 6242 CPUs
(using up to 20 CPUs and 150 cores in parallel). The timeframe was May 15, 2022 to June 1, 2022.

## Data Preprocessing

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.
 
There was no preprocessing that needed to be done.

### Does this dataset collection/processing procedure achieve the motivation for creating the dataset stated in the first section of this datasheet? If not, what are the limitations?
 
Yes, the dataset collection procedure achieves our motivation. See Table 6 for a list of the speedups in NAS experiments achieved when using our dataset.

### Any other comments
 
None.

## Dataset Distribution

### How will the dataset be distributed? (e.g., tarball on website, API, GitHub; does the data have a DOI and is it archived redundantly?)
 
The dataset is on Google Drive, with a DOI.


### When will the dataset be released/first distributed? What license (if any) is it distributed under?
 
The dataset is public as of June 8, 2022, distributed under the Apache License 2.0.

### Are there any copyrights on the data?
 
There are no copyrights on the data.

### Are there any fees or access/export restrictions?
 
There are no fees or restrictions.

### Any other comments?
 
None.

## Dataset Maintenance

### Who is supporting/hosting/maintaining the dataset?
 
The authors of this work are supporting/hosting/maintaining the dataset.

### Will the dataset be updated? If so, how often and by whom?
 
If new NAS benchmarks are created in the NAS research community, the authors of this work may update NAS-Bench-Suite-Zero to include ZC proxy values for the new benchmarks.
Similarly, if new ZC proxies are relased, the authors may update NAS-Bench-Suite-Zero to include the new ZC proxies.

### How will updates be communicated? (e.g., mailing list, GitHub)
 
Updates will be communicated on the GitHub README of this project.

### If the dataset becomes obsolete how will this be communicated?
 
If the dataset becomes obsolete, it will be communicated on the GitHub README of this project.


### If others want to extend/augment/build on this dataset, is there a mechanism for them to do so? If so, is there a process for tracking/assessing the quality of those contributions. What is the process for communicating/distributing these contributions to users?
 
Others can create a pull request or raise an issue on GitHub with possible extensions/augmentations to our dataset, which will be approved in a case-by-case basis. For example, an author of a new ZC proxy may create a PR in our codebase with the new ZC proxy, and then we will evaluate the ZC proxy on all architectures in NAS-Bench-Suite-Zero and update the dataset. These updates will again be communicated on the GitHub README.


## Legal and Ethical Considerations

### Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.
 
There was no ethical review process. We note that our dataset was created by simply by running ZC proxy computations on architectures of existing NAS benchmarks, in some cases using publicly available, licensed datasets such as CIFAR-10 or CIFAR-100.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctorpatient confidentiality, data that includes the content of individuals non-public communications)? If so, please provide a description.
 
The dataset does not contain any confidential data.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.
 
None of the data might be offensive, insulting, threatening, or otherwise cause anxiety.

### Does the dataset relate to people? If not, you may skip the remaining questions in this section.
 
The dataset does not relate to people.

### Any other comments?
 
None.








