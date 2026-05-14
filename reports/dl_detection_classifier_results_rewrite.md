# Kết quả học sâu cho bài toán phát hiện lỗi trực tiếp

## Phạm vi kết quả

Phần này trình bày kết quả thực nghiệm của các mô hình học sâu trong bài toán phát hiện lỗi trực tiếp trên chuỗi dữ liệu SCADA. Nguồn kết quả được sử dụng là thư mục:

`D:\Final Project\scada-fault-prediction\results\sequence_training_results_detection_focal_all_models`

Đây là thí nghiệm với chế độ nhãn **detection**, không phải thí nghiệm cảnh báo sớm. Trong các tệp dự đoán, `horizon_steps = 0`, nghĩa là nhãn của mỗi cửa sổ được lấy theo trạng thái tại thời điểm cuối cửa sổ. Vì vậy, các kết quả dưới đây cần được hiểu là khả năng nhận diện trạng thái lỗi hiện tại trong một cửa sổ dữ liệu 24 giờ, thay vì dự báo lỗi trong một khoảng thời gian tương lai.

## Thiết lập thực nghiệm

Bốn kiến trúc học sâu được huấn luyện và đánh giá gồm:

- LSTM
- GRU
- CNN-LSTM
- CNN-GRU

Mỗi mẫu đầu vào là một chuỗi SCADA dài 24 giờ. Với chu kỳ lấy mẫu 10 phút, mỗi cửa sổ gồm 144 bước thời gian. Mỗi bước thời gian có 21 đặc trưng đầu vào. Kích thước dữ liệu sau khi tạo cửa sổ được tổng hợp như sau.

| Tập dữ liệu | Số cửa sổ | Kích thước đầu vào |
|---|---:|---|
| Huấn luyện | 161,854 | 144 x 21 |
| Kiểm định | 28,138 | 144 x 21 |
| Kiểm thử | 7,918 | 144 x 21 |

Các mô hình được huấn luyện với learning rate `0.001`, không sử dụng L2 regularization, và dùng focal loss với `gamma = 2.0`, `alpha = 0.75`. Focal loss được chọn vì dữ liệu lỗi trong hệ thống SCADA thường mất cân bằng, trong đó lớp lỗi ít hơn và khó học hơn lớp vận hành bình thường.

Ngưỡng phân loại không được cố định tại 0.5. Thay vào đó, mỗi mô hình chọn ngưỡng tốt nhất từ tập validation thông qua quét ngưỡng theo F1-score (`validation_f1_sweep`). Điều này giúp kết quả phản ánh tốt hơn sự đánh đổi giữa phát hiện lỗi và kiểm soát cảnh báo sai.

## Kết quả ở mức cửa sổ

Đánh giá mức cửa sổ xem mỗi cửa sổ 24 giờ là một mẫu độc lập. Đây là lớp đánh giá trực tiếp nhất đối với mô hình học sâu, vì quá trình huấn luyện và dự đoán đều hoạt động trên từng cửa sổ chuỗi.

| Mô hình | Ngưỡng | Accuracy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.418 | 0.673 | 0.523 | **0.758** | 0.619 | 0.638 | 0.758 |
| GRU | 0.495 | 0.804 | 0.873 | 0.516 | 0.649 | 0.822 | 0.876 |
| CNN-LSTM | 0.245 | **0.834** | 0.815 | 0.679 | **0.741** | **0.840** | **0.882** |
| CNN-GRU | 0.333 | 0.793 | **0.877** | 0.473 | 0.615 | 0.822 | 0.873 |

CNN-LSTM là mô hình tốt nhất ở mức cửa sổ. Mô hình đạt F1-score 0.741, PR-AUC 0.840 và ROC-AUC 0.882. Đây là kết quả cân bằng nhất trong nhóm học sâu: precision đủ cao để hạn chế cảnh báo sai, trong khi recall vẫn giữ ở mức tốt hơn GRU và CNN-GRU.

LSTM có recall cao nhất, đạt 0.758, nhưng precision chỉ đạt 0.523. Điều này cho thấy LSTM có xu hướng cảnh báo nhiều hơn, phát hiện được nhiều cửa sổ lỗi hơn nhưng cũng tạo ra nhiều cảnh báo sai hơn. Ngược lại, GRU và CNN-GRU có precision rất cao, lần lượt 0.873 và 0.877, nhưng recall thấp hơn, tức là hai mô hình này thận trọng hơn và bỏ sót nhiều cửa sổ lỗi hơn.

## Ma trận nhầm lẫn trên tập kiểm thử

Tập kiểm thử có 7,918 cửa sổ, trong đó 2,769 cửa sổ là lỗi và 5,149 cửa sổ là bình thường. Ma trận nhầm lẫn cho thấy rõ cách mỗi mô hình đánh đổi giữa false positive và false negative.

| Mô hình | True Negative | False Positive | False Negative | True Positive |
|---|---:|---:|---:|---:|
| LSTM | 3,231 | 1,918 | 670 | 2,099 |
| GRU | 4,942 | 207 | 1,341 | 1,428 |
| CNN-LSTM | 4,722 | 427 | 888 | 1,881 |
| CNN-GRU | 4,966 | 183 | 1,459 | 1,310 |

LSTM phát hiện được số cửa sổ lỗi nhiều nhất với 2,099 true positive, nhưng đồng thời tạo ra 1,918 false positive. Đây là lý do mô hình có recall cao nhưng precision thấp. GRU và CNN-GRU giảm false positive rất mạnh, chỉ còn 207 và 183, nhưng phải đánh đổi bằng số false negative lớn hơn. CNN-LSTM nằm giữa hai hướng này: false positive thấp hơn LSTM rất nhiều, nhưng vẫn giữ được 1,881 true positive. Vì vậy, CNN-LSTM đạt F1-score cao nhất.

## Kết quả ở mức sự kiện

Bên cạnh đánh giá từng cửa sổ, kết quả còn được tổng hợp ở mức sự kiện bằng cách gom các cửa sổ theo `asset_id` và `sequence_id`. Mức đánh giá này gần hơn với thực tế vận hành: thay vì chỉ hỏi mô hình dự đoán đúng bao nhiêu cửa sổ, ta xem mô hình có phát hiện được cả một sự kiện lỗi hay không.

Tập kiểm thử ở mức sự kiện có 22 sự kiện, gồm 12 sự kiện lỗi và 10 sự kiện bình thường.

| Mô hình | Event Precision | Event Recall | Event F1 | Event ROC-AUC |
|---|---:|---:|---:|---:|
| LSTM | 0.571 | **1.000** | 0.727 | 0.908 |
| GRU | 0.647 | 0.917 | 0.759 | 0.867 |
| CNN-LSTM | 0.632 | **1.000** | 0.774 | **0.950** |
| CNN-GRU | **0.733** | 0.917 | **0.815** | 0.917 |

Ở mức sự kiện, CNN-GRU đạt Event F1 cao nhất với 0.815 nhờ precision tốt hơn. CNN-LSTM và LSTM đạt Event Recall 1.000, nghĩa là cả 12 sự kiện lỗi trong tập kiểm thử đều được phát hiện. CNN-LSTM cũng có Event ROC-AUC cao nhất, đạt 0.950, cho thấy điểm rủi ro của mô hình có khả năng xếp hạng tốt giữa sự kiện lỗi và sự kiện bình thường.

Tuy nhiên, cần thận trọng khi diễn giải event-level vì tập đánh giá chỉ có 22 sự kiện. Kết quả này rất hữu ích để phân tích ý nghĩa vận hành, nhưng không nên dùng một mình để kết luận mô hình tốt nhất. Với báo cáo tổng thể, nên kết hợp cả window-level và event-level.

## Phân tích từng kiến trúc

### LSTM

LSTM là mô hình tuần tự thuần, phù hợp để học phụ thuộc thời gian trong chuỗi SCADA. Trên tập kiểm thử, LSTM đạt recall 0.758, cao nhất ở mức cửa sổ. Điều này cho thấy mô hình nhạy với trạng thái lỗi và có khả năng phát hiện nhiều mẫu bất thường.

Hạn chế chính của LSTM là precision thấp, chỉ đạt 0.523. Mô hình tạo ra 1,918 false positive, cao hơn nhiều so với các kiến trúc còn lại. Vì vậy, LSTM phù hợp với mục tiêu ưu tiên không bỏ sót lỗi, nhưng chưa phù hợp nếu hệ thống cần giảm số lượng cảnh báo sai cho người vận hành.

### GRU

GRU đạt accuracy 0.804 và precision 0.873. So với LSTM, mô hình kiểm soát cảnh báo sai tốt hơn rất nhiều. Số false positive của GRU chỉ là 207, thấp hơn gần 9 lần so với LSTM.

Điểm yếu của GRU là recall chỉ đạt 0.516. Mô hình bỏ sót 1,341 cửa sổ lỗi, cho thấy xu hướng dự đoán bảo thủ. Trong bối cảnh giám sát turbine gió, việc giảm cảnh báo sai là quan trọng, nhưng bỏ sót quá nhiều trạng thái lỗi có thể làm giảm giá trị vận hành của hệ thống.

### CNN-LSTM

CNN-LSTM là mô hình cân bằng nhất và có kết quả tốt nhất ở mức cửa sổ. Mô hình đạt accuracy 0.834, precision 0.815, recall 0.679 và F1-score 0.741. So với LSTM, CNN-LSTM giảm false positive từ 1,918 xuống còn 427, trong khi vẫn phát hiện được 1,881 cửa sổ lỗi. So với GRU, CNN-LSTM có recall cao hơn rõ rệt.

Kết quả này cho thấy việc kết hợp CNN và LSTM là phù hợp với dữ liệu SCADA dạng chuỗi. CNN giúp trích xuất các mẫu biến động cục bộ trong chuỗi cảm biến, còn LSTM học quan hệ theo thời gian sau khi biểu diễn đã được làm giàu. Vì vậy, CNN-LSTM nên được xem là mô hình học sâu chính trong phần kết quả của Chương 5.

### CNN-GRU

CNN-GRU có precision cao nhất ở mức cửa sổ, đạt 0.877, và Event F1 cao nhất, đạt 0.815. Mô hình tạo ra ít false positive nhất, chỉ 183 trường hợp trên tập kiểm thử. Điều này cho thấy CNN-GRU là mô hình thận trọng, phù hợp khi hệ thống yêu cầu mỗi cảnh báo phát ra phải có độ tin cậy cao.

Hạn chế của CNN-GRU là recall ở mức cửa sổ thấp nhất trong bốn mô hình, chỉ đạt 0.473. Mô hình bỏ sót 1,459 cửa sổ lỗi. Do đó, CNN-GRU không phải lựa chọn tốt nhất nếu mục tiêu chính là phát hiện càng nhiều trạng thái lỗi càng tốt.

## So sánh với mô hình baseline

Trong phần baseline của Chương 5, XGBoost đạt accuracy 0.693, precision 0.647, recall 0.807 và F1-score 0.718. Khi so sánh với CNN-LSTM, mô hình học sâu đạt accuracy, precision và F1-score cao hơn:

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| XGBoost baseline | 0.693 | 0.647 | **0.807** | 0.718 |
| CNN-LSTM | **0.834** | **0.815** | 0.679 | **0.741** |

So sánh này cho thấy CNN-LSTM cải thiện rõ về độ chính xác tổng thể và độ tin cậy của cảnh báo. Tuy nhiên, XGBoost vẫn có recall cao hơn. Nói cách khác, XGBoost nhạy hơn trong việc bắt lỗi, còn CNN-LSTM cân bằng hơn giữa phát hiện lỗi và hạn chế cảnh báo sai.

Vì vậy, kết luận hợp lý không phải là học sâu vượt trội tuyệt đối trên mọi tiêu chí, mà là CNN-LSTM tạo ra điểm cân bằng tốt hơn cho hệ thống phát hiện lỗi trực tiếp. Nếu ưu tiên recall tuyệt đối, XGBoost vẫn là một baseline mạnh. Nếu ưu tiên F1-score và precision tốt hơn, CNN-LSTM là lựa chọn nổi bật hơn.

## Kết luận

Từ thí nghiệm `sequence_training_results_detection_focal_all_models`, có thể rút ra ba kết luận chính.

Thứ nhất, CNN-LSTM là mô hình học sâu tốt nhất ở mức cửa sổ, với F1-score 0.741, PR-AUC 0.840 và ROC-AUC 0.882. Đây là mô hình cân bằng nhất giữa precision và recall.

Thứ hai, CNN-GRU cho kết quả tốt nhất ở mức sự kiện, với Event F1 đạt 0.815 và Event Precision đạt 0.733. Tuy nhiên, recall ở mức cửa sổ của CNN-GRU thấp, nên mô hình này phù hợp hơn với bối cảnh cần cảnh báo ít nhưng chắc chắn.

Thứ ba, kết quả cho thấy bài toán phát hiện lỗi SCADA vẫn có sự đánh đổi rõ giữa bỏ sót lỗi và cảnh báo sai. LSTM và XGBoost nhạy hơn với lỗi, trong khi GRU và CNN-GRU thận trọng hơn. CNN-LSTM là điểm cân bằng tốt nhất trong nhóm học sâu và phù hợp nhất để trình bày như mô hình chính trong phần kết quả học sâu của Chương 5.

