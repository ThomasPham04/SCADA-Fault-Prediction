CARE to Compare: A real-world dataset for anomaly
detection in wind turbine data
Christian G ̈ucka,∗, Cyriana M.A. Roelofsa, Stefan Faulsticha
aFraunhofer IEE, Joseph-Beuys-Straße 8, 34117 Kassel, Germany
Abstract
Anomaly detection plays a crucial role in the field of predictive maintenance
for wind turbines, yet the comparison of different algorithms poses a difficult
task because domain specific public datasets are scarce. Many comparisons of
different approaches either use benchmarks composed of data from many dif-
ferent domains, inaccessible data or one of the few publicly available datasets
which lack detailed information about the faults. Moreover, many publica-
tions highlight a couple of case studies where fault detection was successful.
With this paper we publish a high quality dataset that contains data from
36 wind turbines across 3 different wind farms as well as the most detailed
fault information of any public wind turbine dataset as far as we know. The
new dataset contains 89 years worth of real-world operating data of wind
turbines, distributed across 44 labeled time frames for anomalies that led up
to faults, as well as 51 time series representing normal behavior. Addition-
ally, the quality of training data is ensured by turbine-status-based labels
for each data point. Furthermore, we propose a new scoring method, called
CARE (Coverage, Accuracy, Reliability and Earliness), which takes advan-
tage of the information depth that is present in the dataset to identify a
good all-around anomaly detection model. This score considers the anomaly
detection performance, the ability to recognize normal behavior properly and
the capability to raise as few false alarms as possible while simultaneously
detecting anomalies early.
Keywords: benchmark, anomaly detection, wind turbines, predictive
∗Corresponding author
Email addresses:christian.gueck@iee.fraunhofer.de(Christian G ̈uck),
cyriana.roelofs@iee.fraunhofer.de(Cyriana M.A. Roelofs),
stefan.faulstich@iee.fraunhofer.de(Stefan Faulstich)
Preprint submitted to Elsevier April 15, 2024
arXiv:2404.10320v2 [cs.LG] 18 Apr 2024
maintenance, fault detection, condition monitoring

Introduction
Wind energy plays a crucial role in the transition to renewable energy, but
monitoring and maintaining wind farms and turbines is a costly challenge.
These farms are often located in regions with challenging weather conditions,
leading to complex operating conditions and increased risk of unexpected fail-
ures and downtime. Over the past decade, various approaches for condition
monitoring, many of which focus on early fault detection using Supervisory
Control and Data Acquisition (SCADA) data, have been investigated [1–3].
A common method to detect component failures early is anomaly detec-
tion (AD), which identifies outliers or other anomalous patterns in the data.
In the context of wind turbines (WTs), most AD techniques utilize data from
the SCADA system, failure logs, vibration data and occasionally status and
maintenance logs [4]. This paper specifically focuses on AD models based on
SCADA data which are validated using additional failure information.
While there have been several benchmarks [5, 6], reviews [7] and com-
parisons [8] of general AD-algorithms, most of them use data from a wide
variety of domains like spacecraft, medical applications and IT-related data.
However, efforts on wind energy specific AD are usually based on non-public
inaccessible data. For example [9, 10] use inaccessible data from wind farms
that are located in China, and [11–13] use data from anonymized offshore
wind farms. These are 5 recently published examples, which lack the ability
for meaningful comparisons between the proposed fault detection algorithms.
Also, inaccessible data prevents reproducibility of presented results. Some
studies have used public wind energy datasets (for example [4, 14–20]), but
they lack comprehensive information about anomalies or component faults.
The lack of extensive public datasets with both SCADA time series and fail-
ure information is a significant limitation in the field of WT SCADA data
analysis. To enable meaningful comparisons between AD algorithms in the
wind energy domain, new public benchmark datasets are necessary.
The main contribution of our work is the publication of the most exten-
sive WT SCADA dataset^1 for AD yet. This includes high dimensional data

(^1) The data can be found on “Fordatis”: http://dx.doi.org/10.24406/fordatis/
and on “Zenodo”:https://zenodo.org/doi/10.5281/zenodo.

from multiple wind farms, information about the WT status at all times,
labeled anomalies with annotated starts and ends and additional fault de-
scriptions. Because the data stems from real-world operating wind farms it
had to be anonymized with the focus on minimizing the loss of useful in-
formation and maximizing the meaningfulness of this dataset for AD and
predictive maintenance.
In addition to the dataset we also provide a sophisticated score, the
Coverage Accuracy Reliability Earliness (CARE)-score, for evaluating AD-
algorithms on this and similar datasets. This score takes into account four key
aspects of a high-quality AD model for predictive maintenance. In combina-
tion with the dataset this score provides the possibility to compare a variety
of different AD-algorithms, from unsupervised to semi-supervised techniques,
designed for early fault detection in WT.
The content of this paper divides into the following sections. At first
we give an overview about the related work in section 2. After that we
introduce the dataset in section 3 by giving information about the layout,
the requirements we set for the quality of the data, the labeling process and
the anonymization actions that were taken. Following this we provide our
scoring idea together with a mini-benchmark of a few selected AD-algorithms
in section 4. Finally a summary concludes this work in section 5.

Related work
In the field of AD benchmark data, many studies focus on dataset com-
positions from various different domains and use cases. Many benchmark
datasets also include a mix of artificial data and data from real-world appli-
cations. While [5, 7, 21, 22] study a wide spectrum of different AD algorithms
for a broad collection of data types, there are also several AD benchmarks

Nomenclature:

WT wind turbine

AD anomaly detection

NBM normal behaviour model

SCADA Supervisory Control and Data Acquisition

WS weighted score

Acc accuracy-score

CARE Coverage Accuracy Reliability Earliness

NN neural network

AE autoencoder

RE reconstruction error

AUC area under the curve

ROC receiver operating characteristic curve

which focus on time series data. One example of such benchmarks is the
widely used and cited Numenta benchmark [6], which provides a collection
of datasets. A more recent and more comprehensive evaluation of AD in time
series is found in [8], where over 71 algorithms were evaluated on more than
900 time series.
In the scope of AD for WTs the time series datasets mentioned above
are usually too broad to be used for evaluation in this specific context. Also,
many are either univariate or synthetic time series and therefore not applica-
ble to AD in SCADA-data. Unfortunately, most domain-specific evaluations
are conducted on inaccessible data that were provided only for the research
in which they are used. As mentioned in the introduction, there are plenty
examples of studies which use such datasets.
There are only a handful of open datasets containing WT SCADA-data.
The studies [23, 24] give an overview about existing datasets for WTs. Addi-
tionally the Git-repository [25] summarizes some currently existing datasets
although some of the listed datasets, such as the SCADA-data of the ENGIE
wind farm “La Haute Borne”, are not available anymore. The most relevant
public dataset in the context of AD for early fault detection is provided by
the EDP open data platform [26], since it is, as far as the authors know,
the only one containing information about WT faults in addition to the
SCADA-data. The faults are provided in form of a start timestamp for some
turbine faults. This dataset was used in the fault detection challenge “hack
the wind” [27] hosted by EDP which is mentioned and evaluated together
with the “WeDoWind”-challenge [28] in [20] and [19]. These challenges fo-
cus in particular on the evaluation of AD-algorithms based on maintenance
cost and potential savings that could be achieved through predictive main-
tenance. Furthermore, several studies on fault detection have used this data
[4, 15, 17, 18]. Although the EDP-dataset is widely used, its level of de-
tail regarding the fault information is small, especially in comparison to the
inaccessible datasets mentioned before.
The lack of publicly available SCADA-datasets of WTs is also acknowl-
edged in [3], which highlights that it is a constraint in the progress of WT
SCADA applications. Additionally, the absence of publicly available datasets
containing real-world anomalies is recognized as a significant obstacle in the
development of AD in general, as it may not adequately reflect the perfor-
mance of methods in real-world applications [7, 22].
But there is not only need for additional domain specific public datasets,
the data quality and the level of detail also plays an important role. As

pointed out by [29] many AD benchmark datasets suffer from flaws that
limit their significance. The main flaws are defined as the flaw of “Triviality”,
“Unrealistic Anomaly Density”, “Mislabeled Ground Truth” and the “Run-
to-Failure Bias”. In the case of publicly available wind SCADA datasets, one
common issue is the absence of labels, particularly regarding fault informa-
tion.

Considering the flaws in datasets, scoring for AD algorithms poses a dif-
ficult challenge. Many studies utilize standard classification metrics such as
accuracy, precision, recall, or the area under the curve (AUC) of the receiver
operating characteristic curve (ROC) [5, 11, 13, 30].
While it is possible to evaluate AD algorithms using the AUC-ROC score
for all possible thresholds, for most practical applications it is much more
useful to have a high F-Score, or a related score. In [31] several variants
of F-scores are compared. The standard pointwise F-Score is the simplest,
but for most use cases, the interest lies in detecting anomaly events, i.e., a
continuous set of anomalous time points, rather than individual time points.
A composite F-score is introduced, a modification of the classic F-score that
takes into account anomaly events through event-wise recall.
Another approach, presented in [32], modifies the classic AUC-ROC met-
ric by generalizing the concept of the ROC to the Preceding-Window-ROC,
thereby adjusting the measure to better fit AD evaluations on time series
data from an event-based perspective.
Finally, the Numenta Benchmark [6] defines a score that is supposed to
measure the performance of more general AD models for time series data
across different domains. The score is based on 5 key-aspects of a good
AD model: “detection of all anomalies”, “early detection of anomalies”, “no
false alarms”, “uses only real time data” and “automation across all different
datasets”.

Based on the provided overview of related work, this paper contributes
to the progress of AD for predictive maintenance on WTs by introducing
a new public dataset that offers more detailed information about turbine
faults and associated anomalies. Furthermore, the new dataset addresses the
flaws identified by [29], although the potential for mislabeled ground truth
cannot be completely eliminated in this context, as the start of anomalous
behavior is often unclear. The flaw of triviality is tackled by the inclusion
of complex anomalies from real-world WTs based on feedback of the wind

farm operators. Additionally, the proposed CARE-score, which differs from
standard classification metrics, draws inspiration from the first three key
aspects of the Numenta score and the composite F-Score from [31], while
distinct adaptions and further developments have been made to better fit
the specific use case of AD for predictive maintenance on WTs.

Data
In this section, we describe the new dataset provided with this paper.
First, we discuss the requirements for a good dataset for AD in WTs in
section 3.1. Then, we provide an overview of the data published in section
3.2, including general statistics such as the number of anomalous events and
features, as well as data quality. In section 3.3, we explain the process of
labeling each time series and datapoint, and in section 3.4 the anonymization
process is described.

3.1. Data requirements

During the process of selecting data for this benchmark dataset, seven re-
quirements were defined to ensure the quality and significance of comparisons
of AD algorithms for AD in WTs. The requirements are as follows:

The dataset must contain as many anomaly events as possible.
The dataset must contain different wind farms.
The dataset must contain different fault types.
The dataset must be balanced, i.e. contain enough prediction data
representing normal behavior.
Every sub-dataset must contain enough normal behavior data in the
intended training time frame. If at least 2/3 of the training data are
normal behavior data we define the sub-dataset to be sufficient.
Every sub-dataset must contain at least one whole year worth of data,
to be able to learn seasonality-related effects.
Every anomaly must have an assigned start timestamp. The anomaly
end is the start of a turbine fault.
While requirements 1 to 3 are necessary to test the generalization ability
of AD algorithms, requirement 4 enables tests for the ability to learn nor-
mal behavior effectively. This is particularly important for the evaluation of

normal behaviour models (NBMs). Additionally, requirements 5 and 6 en-
sures the quality of the training data, to guarantee an NBM can be trained.
Finally, requirement 7 allows for the evaluation of AD models using classifica-
tion measures. These requirements ensure that the dataset is of high-quality,
comprehensive and balanced to train a proper NBM, with detailed labels to
validate the model. All these properties are also relevant for the definition
of the score introduced in section 4.1.

3.2. Dataset

The data consists of 95 datasets, containing 89 years of SCADA time
series distributed across 36 different WTs from the three wind farms A, B
and C. The data for Wind farm A is based on the earlier mentioned EDP-
data [26], and consists of 5 WTs of an onshore wind farm in Portugal. From
this data 22 datasets were selected to be included in this data collection. The
other two wind farms are offshore wind farms located in Germany. All three
datasets were anonymized as described in section 3.4. The overall dataset
is balanced, as 44 out the 95 datasets contain a labeled anomaly event and
the other 51 datasets represent normal behavior. Each dataset is provided in
form of a csv-file with columns defining the features and rows representing
the data points of the time series.
The datasets consist of SCADA time series data for each turbine, with a
resolution of 10 minutes. Each dataset includes one year worth of data for
training a model, as well as 4 to 98 days of prediction data.
The prediction data is divided into an event time frame, with varying
amounts of padding data before and after the event. This padding is used
to prevent guessing the event label (“anomaly” or “normal”) based on the
amount of prediction data.
The number of features in the datasets varies depending on the wind
farm. Wind farm A has 86 features, wind farm B has 257 features, and wind
farm C has 957 features. In addition to the sensor data features, each time
series includes 5 descriptive features: a row ID and a timestamp, an asset
ID that identifies the WT, a “traintest” column indicating whether the row
belongs to the training or prediction data, and a status-ID indicating the
turbine status at the timestamp.
The remaining features represent sensor measurements. For each sensor,
the 10-minute average value is available. Some sensors also have additional
information in the form of 10-minute minimum, maximum, and standard
deviation values. The original sensor names have been replaced in order to

anonymize the data, as described in section 3.4. Only features that describe
power, reactive power or wind speed are recognizable by their name. To
accommodate for the loss in information, additional descriptions are provided
for every sensor. These descriptions include a brief text, the unit of the
sensor as well as boolean indicators that imply whether the sensor represents
a regular sensor signal, a counter or an angle. The most important statistics
of the data are summarized in table 3.2. The rows “Anomaly events” and
“Normal behavior” describe the number of datasets containing an anomaly
event and without anomalies respectively.
Regarding the data quality there are two challenges. The data for wind
farm B and C was provided by the operator with 0-values replacing all miss-
ing values, so large amounts of consecutive 0-values must be treated with
caution. Secondly, note that the status values for wind farm B and C may
be inconsistent; often the status is only logged when it changes, which may
fail if there is a brief communication error. Also, the status values for wind
farm A were derived based on the EDP fault logbook, which only contained
start timestamps of the faults (see section 3.3). It is therefore advisable to
check the power and wind speed values in addition to the status values to
determine whether the turbine has indeed been operating normally.

Wind Farm
A
Wind Farm
B
Wind Farm
C
Overall
Turbines 5 9 22 36
Datasets 22 15 58 95
Anomaly
events
11 6 27 44
Normal
behavior
11 9 31 51
Features 86 257 957 -
Sensors 54 63 238 -
3.3. Data labeling

The data is labeled on two levels. The first level are the so-called event
labels. If a dataset contains an anomaly event inside the prediction time
frame, the dataset is labeled as an anomaly. If this is not the case it is
labeled as normal. The anomaly labels have been determined either based
on direct feedback by the wind farm operators or based on documented faults
in the form of service reports and fault logbooks. The normal labels have

been determined by a combination of feedback of the wind farm operators,
manual inspection of the data and expert knowledge.
For wind farm A all anomaly event starts were defined based on the
available EDP fault logbook which only defines start timestamps for each
fault. Since no further information is available, analysis of the data before
every fault was used to determine possible event starts. The ’true’ anomaly
event starts for wind farm A can differ from the set ones.
For the wind farms B and C all starts of the anomaly events were defined
based on data analysis, feedback of the wind farm operator, service report
documents and expert knowledge. While the true starts of the anomaly
events could potentially differ from the set ones in some cases, it is highly
unlikely that the defined events start too early. If anything, anomaly event
start could be earlier than defined.
The second level of labeling assigns a label to each timestamp of every
dataset. These labels are called status-IDs. For the wind farms B and C
they are derived from the original operating modes that were provided by
the wind farm operators in combination with service report information. For
wind farm A this information was not provided. In this case the status-IDs
were based on the fault information from the logbook provided by EDP. For
each turbine fault the preceding 14 days were marked with the status-ID 4
(fault) and the 3 days after the fault timestamp were marked with the status-
ID 3 (service mode). The time ranges around the turbine fault were set with
the aim in mind to reduce the risk of including anomalous behavior in the
training data. As no information is available on the duration of anomalies
before and after the given faults, the time ranges were chosen conservatively.
The status labels can be used to infer whether a given data point rep-
resents normal WT behavior or not. The status-IDs, their description and
whether we consider the status normal, are found in table 3.3.

Status-ID
Description Considered Normal
0 Normal operation
without limitations
True
1 Derated power
generation with a
power restriction
False
2 Asset is idling and
waits to operate again
True
3 Asset is in service
mode / service team is
at the site
False
4 Asset is down due a
fault or other reasons
False
5 Other operational
states for example
system test, setup, ice
build-up or emergeny
power
False
3.4. Anonymization

Due to confidentiality reasons the data of wind farm B and C was anonymized.
The anoymization includes the removal of all information that can directly
identify the wind farms, such as the name of the wind farm, the original
names of each WT, the turbine type and the location. The wind farm names
were replaced by the generic names ”Wind Farm A”, ”Wind Farm B” and
”Wind Farm C” while the WT names were replaced by randomized asset IDs.
However, the asset-IDs were assigned in a way that makes it still possible to
link different datasets that belong to the same WT.
Additionally, the timestamps of each dataset were shifted by a random
number of years. This preserves the consistency of the seasonal information,
although it does distort the temporal order of the datasets.
Names of the original SCADA-features were replaced by a numeration
of the features. Only features that describe power, reactive power or wind
speed are recognizable by their name. Additionally, power and reactive power
features have been scaled with the rated power of the turbine. This way, it
is still possible to clean and analyse the data using the power curve of the
WT.

All status information were aggregated from the original status data of
the WTs and the name of each status condition was replaced by a number in
combination with a brief description. Wind farms B and C contain detailed
status information while wind farm A only contains status information which
indicate turbine faults.

Anomaly detection evaluation
Evaluation of AD algorithms poses a difficult task. On one hand the
perfect AD should detect all anomalies as soon as possible, without any false
alarms, on the other hand labeling of anomalies and finding proper start and
end times of anomaly events cannot be done perfectly.
Although the ground truth of every AD evaluation is almost certainly
flawed [29], standard classification metrics, like the F-Score, accuracy and
precision, are often used to measure the performance of AD algorithms to
compare them with other algorithms or to show their overall performance
[33]. The F-score in particular is widely used, but it cannot be applied
to evaluate the performance of AD algorithms on normal data since true
negatives are not considered in the F-score. This is one of the reasons why
metrics like the F-score are not suitable for a complete evaluation.
To tackle these problems, we introduce the CARE-score in 4.1 to evaluate
models on the dataset described in 3. The score is composed out of four sub-
scores, each evaluating a key aspect of a good AD model. In addition to that,
we conduct a mini-benchmark 4.2 to showcase the CARE-score and dataset.

4.1. The CARE-Score

In the context of AD for predictive maintenance the performance of mod-
els is often difficult to assess. To address this, we introduce the CARE-score
for evaluating AD in an operational predictive maintenance setting. The
CARE-score focuses on four key aspects that a good AD model for predic-
tive maintenance should excel in, which are:

Coverage: Detection of as many correct anomalies as possible,
Accuracy: Recognition of normal behavior,
Reliability: Few false alarm events,
Earliness: Detection of anomalies before fault gets critical.
The CARE-score consists of four sub-scores, each representing one of
the four aspects mentioned above. The first and fourth sub-scores measure
the pointwise classification performance of an algorithm on datasets where
anomaly events are present. The second sub-score considers the model’s per-
formance on datasets without any anomalous data, i.e. its ability to recog-
nize normal behavior accordingly. The third sub-score assesses classification
performance on aggregated events, by applying an eventwise classification
measure.
Contrary to measures like the AUC-ROC all four sub-scores considered
are threshold-specific performance measures. This is a property that makes
comparisons of algorithms for the sake of AD less significant, but in exchange
it raises the significance of the score in a real-world operative predictive
maintenance setting. As mentioned in [31], operators of wind farms and
other assets are primarily interested in accurately detecting anomalies while
minimizing false alarms. Additionally, this enables comparison of the same
NBM with different threshold methods.

4.1.1. Score Definition
Coverage.In order to measure the coverage - i.e. the classification per-
formance on a time series with anomalies - theFβ-score is used. At first
the prediction time frame of a given anomaly event dataset is filtered. All
data points with an abnormal status-ID according to table 3.3 are ignored.
This is important because these data points are usually very easy to detect.
Moreover, the wind farm operator is already informed about the abnormal
behavior through the status-ID. Thus these data points are irrelevant in the
context of predictive maintenance and they would dilute the score. Now let
gbe the ground truth of all data points with a normal status-ID within the
prediction time frame andpis the corresponding prediction of an AD-model.
TheFβ-score is then defined as

Fβ(g,p) =
(1 +β^2 )·tp(g,p)
(1 +β^2 ·tp(g,p) +β^2 ·fn(g,p) +fp(g,p))
, (1)
wheretp(g,p) is the number of true positives based ongandp,fn(g,p) is
the number of false negatives andfp(g,p) is the number of false positives.
In this case, a value ofβ=^12 is chosen to give more weight to precision than
recall, thereby penalizing excessive false positives.

Accuracy. To measure the performance on datasets that exclusively contain
normal behavior, the accuracy-score (Acc) is used. Since there are no true
positives for the prediction time frames of those datasets,Fβfrom equation
1 would always be 0. With the same reasoning as in the coverage-paragraph
4.1.1 only data points with a normal status-ID are relevant. Letgbe the
ground truth of all data points with a normal status-ID within the prediction
time frame andpis the corresponding prediction. Then Acc is calculated by

Acc(g,p) =
tn(g,p)
fp(g,p) +tn(g,p)
. (2)
wheretn(g,p) is the number of true negatives based ongas well aspand
fp(g,p) is the number of false positives. Note that in contrast to the stan-
dard accuracy, there are no true positives and false negatives, since only
datasets containing normal behavior are considered and data points with an
abnormal status-ID are excluded.

Reliability. False alarms on an event basis are taken into account by the
event basedFβ-score (EFβ). First, each time series prediction needs to be
classified as either ‘anomaly event detected’ or ‘normal behavior’. For this,
we first calculate the maximum ‘criticality’, which is a counter-like mea-
sure. Given the prediction timestamps t 1 ,...,tN, the status information
sti ∈ { 0 , 1 }where 1 corresponds to a normal status and 0 to an abnormal
status and the predictionpti∈{ 0 , 1 }where 1 represents a detected anomaly
and 0 no detected anomaly fori= 1,...,N, the criticality is computed by
algorithm 1.
After calculating the criticality for the entire prediction time frame, the
maximum criticality is compared to a thresholdtc, which we set to 72. The
idea behind this threshold is that in order to reach a criticality of 72 the
algorithm must either detect at least 72 anomalies in a row, which equates
to 12 hours of consecutive anomalies, or even more anomalies in the case of
non-consecutive detected anomalies. Setting the threshold at 72 was found
to be the most appropriate for all 95 time series in this dataset and generally
depends on the length of the time series and the use-case specific definitions
of detected anomaly events. For shorter events, a lower threshold is more
appropriate.
If the threshold is exceeded, the prediction is counted as a detected
anomaly event (i.e. an alarm was raised). If the maximum criticality is

Algorithm 1Criticality Algorithm

crit←[0, 0 ,...,0]∈NN+
fori∈{ 1 ,...,N}do
ifsti= 0then
ifpti= 1then
crit[i]←crit[i−1] + 1
else
crit[i]←max{crit[i−1]− 1 , 0 }
end if
else
crit[i]←crit[i−1]
end if
end for
crit←crit[1 :N]
below 72, the prediction is counted as a normal event prediction (i.e. no
alarm). These event prediction labels are then compared to the true dataset
labels and theFβ-score is calculated as defined in equation 1 whereβ=^12 is
usually chosen to penalize false positives further.

Earliness.Similar to theFβ-score, the second sub-score - the weighted
score (WS) - is also only applied to anomaly events. As a modified version of
the weighted score defined in the Numenta benchmark [6], this score weights
detected anomalies during the beginning of defined anomaly events higher
than detected anomalies at the end of the event time frame. However, instead
of discarding additional detected anomalies within the event time frame, all
detected anomalies will be considered with positive weights. The piecewise
linear function shown in figure 1 is used to weight the predicted timestamps.
In the first half of the event, all detected anomalies are assigned a weight
of 1. In the second half of the event, the weights decrease linearly to 0, as
the detected anomalies become less important to the wind farm operator the
closer they are to the actual turbine fault.
In order to apply this weighting function to anomaly events of different
lengths, the length of each event is used to convert the event timestamps
to the relative position in the interval [0,1], where 0 corresponds to the
beginning of the anomaly event and 1 corresponds to the end of the event,

Figure 1: Weight function of the weighted score (WS) for early anomaly detection.
i.e. the start of a downtime or a fault as detected by other systems. Let the
consecutive timestamps of an anomaly eventabet 1 < t 2 <···< tM while
pa:= (pt 1 ,...,ptN)∈ { 0 , 1 }Ndenotes the corresponding prediction where 1
marks a detected anomaly and 0 means no detected anomaly. The WS of
this anomaly event is then calculated as

WS(pa) =
PM
Pi=1wti·pti
M
i=1wti
(3)
wherewti∈[0,1] is the weight for the timestampti.

CARE. Finally, the CARE score is calculated by combining the four sub-
scores. This is done by calculating the event based scoreEFβand the aver-
agesFβ,WSandAcc. Here,Fβis the arithmetic mean over allFβ-scores of
datasets containing an anomaly event,WSis the arithmetic mean over all
WS of datasets containing an anomaly event andAccis the average over all
Acc of datasets representing normal behavior.
The final CARE score takes two special cases into account. If no anoma-
lies were detected at all, the CARE score will be 0. Also, ifAccfalls below
0 .5, the predictions are worse than uniformly distributed random predictions.
In this case the final score will be equal toAcc. Outside of these two special

cases, the CARE score is defined by a weighted averageWAof all sub-scores:

WA:=
1
P 4
i=1ωi
ω 1 Fβ+ω 2 WS+ω 3 EFβ+ω 4 Acc

, (4)
where we chooseω 1 =ω 2 =ω 3 = 1 andω 4 = 2 in order to weight the normal
datasets to the same degree as datasets containing an anomaly event and
β=^12.
To summarize, the CARE-score is defined by

CARE:=



0 , if no anomalies were detected
Acc, ifAcc < 0. 5
WA, else.
(5)
4.1.2. Score Modification

In some cases it is beneficial to adapt the CARE-score for different use
cases. As the AD-challenges [27] and [28] show maintenance costs for different
turbine faults are often considered when assessing the performance of AD-
models. The CARE-score can be adjusted to take such costs into account by
replacingFβandWSby weighted averages.
Leta 1 ,...,aNbe all datasets containing anomaly events andω:= (ω 1 ,...,ωN)
be the cost-based importance weights of each anomaly. The weighted aver-
ages are then be defined by

Fβω:=
1
PN
i=1ωi
XN
i=
ωi·Fβ(gai,pai) (6)
WSω:=
1
PN
i=1ωi
XN
i=
ωi·WS(pai), (7)
wheregai is the ground truth for the prediction time frame of datasetai
andpaiis the corresponding model prediction. Additionally, the weights in
equation 4 can be altered to better suit the use case.
For the following mini-benchmark section no further modification of weights
are made, since the necessary information about maintenance costs for each
fault in the dataset are not available for the wind farms B and C.

4.2. Mini-Benchmark

For the purpose of using the dataset for benchmarks of AD algorithms
for fault detection in WTs, and showcasing the newly defined CARE-score,

python implementations of an NBM based on an autoencoder (AE) approach
and a simple isolation forest approach are compared to the trivial strategies
“all anomaly”, “all normal” and “random”.

4.2.1. Simple approaches
While “all anomaly” just classifies every timestamp as an anomaly and
“all normal” does the opposite, “random” assigns the prediction for every
timestamp independently based on a 50/50 probability. The slightly more
complex isolation forest approach uses the implementation from the python
package “sklearn” [34] with “nestimators”=100 and “contamination”=0.09,
as well as a principal component analysis in order to reduce the dimension-
ality of the input data, such that 99% of the variance is kept. All hyperpa-
rameters were selected by manual tests.

4.2.2. Autoencoder approach
The AE NBM is a more sophisticated approach. The model used is a
further developed version of the AD procedure described in [35]. This model
consists of an AE model trained on data representing normal behavior and a
calibrated threshold to detect anomalies. The AE models for each wind farm
contain 3 to 5 hidden layers and are optimized using the Adam algorithm.
The hyperparameters, such as the number of units in the hidden layers, the
learning rate and the amount of noise to regularize the AE, were adjusted
using the python package “Optuna” [36]. An overview of the model hyper-
parameters for the baseline models is provided in table 4.2.2. The AE is
trained on 75% of the normal training data while 25% of the data are ran-
domly selected for validation. The training lasts at most 200 epochs with an
early stopping option, which is triggered if the L2-norm of the reconstruc-
tion error on the validation data does not decrease for 3 consecutive epochs.
Based on a calibrated threshold, predicted timestamps are assigned the label
“anomaly” if the L2-norm of the corresponding reconstruction error exceeds
the threshold, otherwise they will receive the label “normal”.
The calibration of the threshold differs depending on the wind farm. For
the wind farms A and B an adaptive threshold is used, inspired by the work in
[37] and [30]. Here, a neural network (NN) regression model is used to learn
the mapping of the AE input data to the L2-norm of reconstruction errors
(REs). The NN consists of 3 layers with around 20 to 40 units in the hidden
layer, ReLU activations and the Adam algorithm is used for optimization. For
training the same data which the AE is validated on is used, i.e. the part of

the validation data representing normal behavior. The training lasts at most
300 epochs with the same early stopping mechanism as in the AE training.
During prediction, the new input data is evaluated by the NN and provides
an expected REε. This expected RE is increased by adding the sensitivity
parameterγ∈[0,∞] and then compared to the actual RE of the AE model.
While parameterγhas to be optimized for each wind farm separately, values
from the interval [0. 2 , 0 .4] seem to be a good fit for the provided datasets. If
the actual RE is larger thanε+γ, the corresponding timestamp is detected as
an “anomaly”. For the determination of the optimal number of units in the
hidden layer of the NN and the value forγa hyperparameter optimization
with “Optuna” is used. The final hyperparameter values of the thresholds
used for the benchmark can be found in table 4.2.2.
For wind farm C, a fixed threshold is calibrated. For this, the L2-norm
of the reconstruction errors (anomaly score) of all AE validation data is
computed. Afterwards, the constant threshold is selected by iterating over
the calculated anomaly score values and choosing the value that maximizes
theF 12 -score based the ground truth defined by the normal behavior labels,
that can be derived from the status-IDs as shown in table 3.3.

Wind Farm A Wind Farm B Wind Farm C
# units in
hidden layers
44, 25, 4, 25, 44 40, 15, 40 133, 83, 20, 83,
133
learning rate 0.0018 0.003 0.
noise 0.06 0 0
batch size 64 64 64
threshold adaptive adaptive max.F 12 -score
threshold NN
hidden layers
23 36 -
γ 0.344 0.234 -
Finally the AE NBM is supplemented with an additional data filter. In
order to remove potentially implausible data from the training data of the AE
a status based on wind speed and power enhances the normal behavior labels
given by the turbine operational status from table 3.3. For the determination
of the new status information timestamps are marked as not normal if the
wind speed is within the normal operation range of the WT and the power
is close or equal to 0.

Figure 2: Sub-scores of CARE-score over all 95 sub-datasets for a few selected approaches.

4.2.3. Scoring of the approaches

At first the four sub-scores of the CARE-score are evaluated for all five
approaches. The results are visualized in figure 2. While “all anomaly”
obviously performs well in detecting anomalies, it of course has the worst
possible accuracy on normal data. The opposite is of course the case for
“all normal”. The isolation forest behaves similar to “all anomaly”, since it
detects a lot of anomalies but performed very poorly in recognizing normal
behavior. Finally the AE approach has a high accuracy on normal data and
a good performance on the event basedF 12 -score but it suffers a little bit
from an overall low number of detected anomalies.
When it comes to the final CARE-score shown in figure 3, the trivial
strategies “all anomaly” and “all normal” both get the score 0 because they
run into the special cases described at the end of section 4.1. The strategy
“random” gets the CARE-score of 0.5 and sets the lower bound to beat
for good any anomaly detection algorithms. The isolation forest approach
does not outperform that threshold since it is not able to recognize normal
behavior appropriately with the chosen parameter configuration. With a
score of 0.66 the AE approach represents a good anomaly detector since it
is able to detect anomalies while also recognizing normal behavior very well
and having a good classification performance on aggregated events.

Figure 3: CARE-score benchmark over all 95 sub-datasets for a few selected approaches.

Summary
With the purpose of reducing the limitations that come with the lack of
public data for WT AD benchmarks, a new dataset was published. Composed
out of multiple WTs across 3 wind farms the dataset shows greater detail in
anomaly labels and additional information than datasets that are currently
available. By formulating requirements for benchmark datasets for AD in
WTs, the data quality and the ability to test for generalization of AD models
were ensured. The balanced nature of the dataset, with similar amounts of
anomalous data and examples of normal behavior, allows for more detailed
and meaningful comparison studies of AD algorithms.
Furthermore, we proposed an evaluation method, the CARE-score, that
fully uses the informational depth the dataset provided. By considering the
four key aspects of a good AD-model - detecting many anomalies, early
detection, few false alarms and correct recognition of normal behavior - the
CARE-score provides a measure for the all-around performance of a good
AD model for predictive maintenance on WTs.
To demonstrate the combination of dataset and CARE-score, a ‘mini-
benchmark’ was conducted. A sophisticated AD algorithm was compared to
the popular and simpler isolation forest and 3 trivial strategies. This eval-
uation shows the importance of not neglecting normal behavior recognition
while trying to detect as many anomalies as possible, as it was the case for
the isolation forest approach.
As subject to future research the provided dataset and scoring method
can be used to compare a wide range of WT AD models on an equal and
transparent basis in order to push the progress in this field further and find
good AD algorithms.
To further enhance the development of benchmark datasets in the field of
WT AD, the authors strongly encourage others to share their data. Increas-
ing the availability of high-quality benchmark datasets will facilitate more
comprehensive and rigorous evaluations of AD models in this domain.

Acknowledgements. The development of methods presented was funded by
the German Federal Ministry for Economic Affairs and Climate Action (BMWK).

References

[1] J. Tautz-Weinert, S. J. Watson, Using SCADA data for wind turbine
condition monitoring – a review, IET Renewable Power Generation
11 (4) (2017) 382–394.doi:10.1049/iet-rpg.2016.0248.
URL https://onlinelibrary.wiley.com/doi/10.1049/iet-rpg.
2016.0248
[2] G. Helbing, M. Ritter, Deep Learning for fault detection in wind tur-
bines, Renewable and Sustainable Energy Reviews 98 (2018) 189–198.
doi:10.1016/j.rser.2018.09.012.
URL https://www.sciencedirect.com/science/article/pii/
S1364032118306610

[3] R. Pandit, J. Wang, A comprehensive review on enhancing wind turbine
applications with advanced SCADA data analytics and practical in-
sights, IET Renewable Power Generation 18 (4) (2024) 722–742, eprint:
https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/rpg2.12920.
doi:https://doi.org/10.1049/rpg2.12920.
URL https://ietresearch.onlinelibrary.wiley.com/doi/abs/
10.1049/rpg2.12920

[4] E. Latiffianti, S. Sheng, Y. Ding, Wind Turbine Gearbox Failure Detec-
tion Through Cumulative Sum of Multivariate Time Series Data, Fron-
tiers in Energy Research 10 (2022).doi:10.3389/fenrg.2022.904622.
URL https://www.frontiersin.org/articles/10.3389/fenrg.
2022.904622

[5] S. Han, X. Hu, H. Huang, M. Jiang, Y. Zhao, ADBench: Anomaly
Detection Benchmark, in: S. Koyejo, S. Mohamed, A. Agarwal,
D. Belgrave, K. Cho, A. Oh (Eds.), Advances in Neural Information
Processing Systems, Vol. 35, Curran Associates, Inc., 2022, pp. 32142–

URL https://proceedings.neurips.cc/paper_files/paper/2022/
file/cf93972b116ca5268827d575f2cc226b-Paper-Datasets_and_
Benchmarks.pdf

[6] A. Lavin, S. Ahmad, Evaluating real-time anomaly detection algorithms

the numenta anomaly benchmark, in: 2015 IEEE 14th International
Conference on Machine Learning and Applications (ICMLA), 2015, pp.
38–44.doi:10.1109/ICMLA.2015.141.
[7] G. Pang, C. Shen, L. Cao, A. V. D. Hengel, Deep Learning for Anomaly
Detection: A Review, ACM Computing Surveys 54 (2) (2022) 1–38.

doi:10.1145/3439950.
URLhttps://dl.acm.org/doi/10.1145/3439950
[8] S. Schmidl, P. Wenig, T. Papenbrock, Anomaly detection in time series:
a comprehensive evaluation, Proc. VLDB Endow. 15 (9) (2022) 1779–
1797, publisher: VLDB Endowment.doi:10.14778/3538598.3538602.
URLhttps://doi.org/10.14778/3538598.3538602
[9] C. Zhang, D. Hu, T. Yang, Research of artificial intelligence operations
for wind turbines considering anomaly detection, root cause analysis,
and incremental training, Reliability Engineering & System Safety 241
(2024) 109634.doi:https://doi.org/10.1016/j.ress.2023.109634.
URL https://www.sciencedirect.com/science/article/pii/
S0951832023005483
[10] Yang Luoxiao, Zhang Zijun, A Conditional Convolutional Autoencoder-
Based Method for Monitoring Wind Turbine Blade Breakages, IEEE
Transactions on Industrial Informatics 17 (9) (2021) 6390–6398. doi:
10.1109/TII.2020.3011441.

[11] R. Morrison, X. Liu, Z. Lin, Anomaly detection in wind turbine SCADA
data for power curve cleaning, Renewable Energy 184 (2022) 473–486.
doi:https://doi.org/10.1016/j.renene.2021.11.118.
URL https://www.sciencedirect.com/science/article/pii/
S0960148121017134

[12] L. Schr ̈oder, N. K. Dimitrov, D. R. Verelst, J. A. Sørensen, Using
Transfer Learning to Build Physics-Informed Machine Learning Mod-
els for Improved Wind Farm Monitoring, Energies 15 (2) (2022). doi:
10.3390/en15020558.
URLhttps://www.mdpi.com/1996-1073/15/2/558

[13] C. McKinnon, J. Carroll, A. McDonald, S. Koukoura, D. Infield, C. Sor-
aghan, Comparison of New Anomaly Detection Technique for Wind
Turbine Condition Monitoring Using Gearbox SCADA Data, Energies
13 (19) (2020).doi:10.3390/en13195152.
URLhttps://www.mdpi.com/1996-1073/13/19/5152

[14] X. Jia, Y. Han, Y. Li, Y. Sang, G. Zhang, Condition monitoring and per-
formance forecasting of wind turbines based on denoising autoencoder

and novel convolutional neural networks, Energy Reports 7 (2021) 6354–
6365.doi:10.1016/j.egyr.2021.09.080.
[15] F. P. G. de S ́a, D. N. Brand ̃ao, E. Ogasawara, R. d. C. Coutinho, R. F.
Toso, Wind Turbine Fault Detection: A Semi-Supervised Learning Ap-
proach With Automatic Evolutionary Feature Selection, in: 2020 Inter-
national Conference on Systems, Signals and Image Processing (IWS-
SIP), 2020, pp. 323–328.doi:10.1109/IWSSIP48289.2020.9145244.

[16] W. Udo, Y. Muhammad, Data-Driven Predictive Maintenance of Wind
Turbine Based on SCADA Data, IEEE Access 9 (2021) 162370–162388.
doi:10.1109/ACCESS.2021.3132684.

[17] Z. Tang, X. Shi, H. Zou, Y. Zhu, Y. Yang, Y. Zhang, J. He, Fault
Diagnosis of Wind Turbine Generators Based on Stacking Integration
Algorithm and Adaptive Threshold, Sensors 23 (13) (2023). doi:10.
3390/s23136198.
URLhttps://www.mdpi.com/1424-8220/23/13/6198

[18] M. Jankauskas, A. Serackis, M.Sapurov, R. Pomarnacki, A. Baskys,ˇ
V. K. Hyunh, T. Vaimann, J. Zakis, Exploring the Limits of Early Pre-
dictive Maintenance in Wind Turbines Applying an Anomaly Detection
Technique, Sensors 23 (12) (2023).doi:10.3390/s23125695.
URLhttps://www.mdpi.com/1424-8220/23/12/5695

[19] S. Barber, U. Izagirre, O. Serradilla, J. Olaizola, E. Zugasti, J. I. Aizpu-
rua, A. E. Milani, F. Sehnke, Y. Sakagami, C. Henderson, Best Practice
Data Sharing Guidelines for Wind Turbine Fault Detection Model Eval-
uation, Energies 16 (8) (2023).doi:10.3390/en16083567.
URLhttps://www.mdpi.com/1996-1073/16/8/3567

[20] S. Barber, L. A. M. Lima, Y. Sakagami, J. Quick, E. Latiffianti, Y. Liu,
R. Ferrari, S. Letzgus, X. Zhang, F. Hammer, Enabling Co-Innovation
for a Successful Digital Transformation in Wind Energy Using a New
Digital Ecosystem and a Fault Detection Case Study, Energies 15 (15)
(2022).doi:10.3390/en15155638.
URLhttps://www.mdpi.com/1996-1073/15/15/5638

[21] A. B. Nassif, M. A. Talib, Q. Nasir, F. M. Dakalbab, Machine Learning
for Anomaly Detection: A Systematic Review, IEEE Access 9 (2021)
78658–78700.doi:10.1109/ACCESS.2021.3083060.

[22] L. Ruff, J. R. Kauffmann, R. A. Vandermeulen, G. Montavon,
W. Samek, M. Kloft, T. G. Dietterich, K.-R. Muller, A Unifying Re-
view of Deep and Shallow Anomaly Detection, Proceedings of the IEEE
109 (5) (2021) 756–795.doi:10.1109/JPROC.2021.3052449.
URLhttps://ieeexplore.ieee.org/document/9347460/

[23] N. Effenberger, N. Ludwig, A collection and catego-
rization of open-source wind and wind power datasets,
Wind Energy 25 (10) (2022) 1659–1683, eprint:
https://onlinelibrary.wiley.com/doi/pdf/10.1002/we.2766. doi:
https://doi.org/10.1002/we.2766.
URL https://onlinelibrary.wiley.com/doi/abs/10.1002/we.
2766

[24] D. Menezes, M. Mendes, J. A. Almeida, T. Farinha, Wind Farm and
Resource Datasets: A Comprehensive Survey and Overview, Energies
13 (18) (2020).doi:10.3390/en13184702.
URLhttps://www.mdpi.com/1996-1073/13/18/4702

[25] S. Letzgus, Wind Turbine SCADA open data, [Online; accessed
13-03-2024] (2023).
URL https://github.com/sltzgs/Wind_Turbine_SCADA_open_
data?tab=readme-ov-file

[26] EDP Inova ̧c ̃ao,EDPRWind Farm Open Data: Wind Turbine SCADA
signals and historical failure logbook from 2016 and 2017 (2018).
URLhttps://www.edp.com/en/innovation/open-data/data

[27] EDP Inova ̧c ̃ao, Hack the Wind: Wind Turbine Failures Detection
(2018).
URL https://www.edp.com/en/innovation/open-data/reuses/
hack-the-wind

[28] Eastern Switzerland University of Applied Sciences, Wo do Wind: EDP
Challenges space (2021).
URLhttps://www.wedowind.ch/spaces/edp-challenges-space

[29] R. Wu, E. J. Keogh, Current time series anomaly detection benchmarks
are flawed and are creating the illusion of progress, IEEE Transactions

on Knowledge and Data Engineering 35 (3) (2023) 2421–2429. doi:
10.1109/TKDE.2021.3112126.
[30] H. Chen, H. Liu, X. Chu, Q. Liu, D. Xue, Anomaly detection and
critical SCADA parameters identification for wind turbines based on
LSTM-AE neural network, Renewable Energy 172 (2021) 829–840.
doi:https://doi.org/10.1016/j.renene.2021.03.078.
URL https://www.sciencedirect.com/science/article/pii/
S0960148121004341

[31] A. Garg, W. Zhang, J. Samaran, R. Savitha, C.-S. Foo, An evaluation
of anomaly detection and diagnosis in multivariate time series, IEEE
Transactions on Neural Networks and Learning Systems 33 (6) (2022)
2508–2517.doi:10.1109/TNNLS.2021.3105827.

[32] J. Carrasco, D. L ́opez, I. Aguilera-Martos, D. Garc ́ıa-Gil,
I. Markova, M. Garc ́ıa-Barzana, M. Arias-Rodil, J. Luengo,
F. Herrera, Anomaly detection in predictive maintenance: A
new evaluation framework for temporal unsupervised anomaly
detection algorithms, Neurocomputing 462 (2021) 440–452.
doi:https://doi.org/10.1016/j.neucom.2021.07.095.
URL https://www.sciencedirect.com/science/article/pii/
S0925231221011826

[33] A. Stetco, F. Dinmohammadi, X. Zhao, V. Robu, D. Flynn, M. Barnes,
J. Keane, G. Nenadic, Machine learning methods for wind turbine
condition monitoring: A review, Renewable Energy 133 (2019) 620–635.
doi:10.1016/j.renene.2018.10.047.
URL https://linkinghub.elsevier.com/retrieve/pii/
S096014811831231X

[34] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion,
O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vander-
plas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, E. Duchesnay,
Scikit-learn: Machine learning in Python, Journal of Machine Learning
Research 12 (2011) 2825–2830.

[35] C. M. Roelofs, M.-A. Lutz, S. Faulstich, S. Vogt, Autoencoder-based
anomaly root cause analysis for wind turbines, Energy and AI 4 (2021)
100065.doi:10.1016/j.egyai.2021.100065.

[36] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A next-
generation hyperparameter optimization framework, in: Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Dis-
covery and Data Mining, 2019, p. 2623–2631.

[37] H. Zhao, H. Liu, W. Hu, X. Yan, Anomaly detection and fault analysis
of wind turbine components based on deep learning network, Renewable
Energy 127 (2018) 825–834.doi:10.1016/j.renene.2018.05.024.
URL https://www.sciencedirect.com/science/article/pii/
S0960148118305457