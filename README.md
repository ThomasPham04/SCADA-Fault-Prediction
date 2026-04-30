# Phát triển mô hình học sâu để dự báo lỗi thiết bị dựa trên dữ liệu SCADA

## Dàn ý README phục vụ viết khóa luận tốt nghiệp

Tài liệu này được viết lại theo hướng học thuật để làm sườn triển khai khóa luận/capstone từ chính codebase hiện tại. Bố cục tham khảo cách tổ chức vấn đề trong bài báo *Fault detection of wind turbine based on SCADA data analysis using CNN and LSTM with attention mechanism*, nhưng nội dung dưới đây mô tả trung thực pipeline chính hiện nay: chuẩn bị combined-sequence CSV, tạo cửa sổ theo `asset_id + sequence_id`, huấn luyện sequence classifiers và sequence autoencoders, rồi báo cáo kết quả theo window-level và event-level.

> Lưu ý học thuật: repository này **không phải** bản tái hiện đầy đủ mô hình CNN-LSTM với attention của bài báo tham chiếu. Khi viết báo cáo chính thức, bài báo nên được dùng ở phần tổng quan nghiên cứu, cơ sở tham khảo kiến trúc và định hướng mở rộng.

---

## Thông tin đề tài

- **Tên đề tài tiếng Việt:** Phát triển mô hình học sâu để dự báo lỗi thiết bị dựa trên dữ liệu SCADA.
- **Tên đề tài tiếng Anh:** Development of Deep Learning Models for Equipment Fault Prediction using SCADA Data.
- **Bài toán chính:** phát hiện sớm bất thường và dự báo lỗi thiết bị từ chuỗi thời gian đa biến thu thập bởi hệ thống SCADA.
- **Ngữ cảnh ứng dụng:** giám sát vận hành tua-bin gió, giảm thời gian dừng máy, hỗ trợ bảo trì dự đoán.
- **Phạm vi repo hiện tại:** tập trung vào bộ dữ liệu CARE to Compare, ưu tiên combined-sequence pipeline cho Wind Farm A trong cấu hình chính, và giữ Wind Farm B/C như nguồn dữ liệu mở rộng để phân tích.
- **Đóng góp kỳ vọng của khóa luận:** xây dựng pipeline tiền xử lý thống nhất, thiết kế mô hình phát hiện lỗi trên dữ liệu SCADA, đánh giá bằng các chỉ số thực nghiệm, và đề xuất hướng mở rộng sang các kiến trúc sâu hơn như CNN-LSTM-attention hoặc Transformer.

---

## MỞ ĐẦU

### 1. Lý do chọn đề tài

- **Cần viết:** nhu cầu phát hiện sớm lỗi trong hệ thống công nghiệp, đặc biệt là tua-bin gió, nơi lỗi thường tích lũy dần và chi phí bảo trì đột xuất rất lớn.
- **Nên nhấn mạnh:** SCADA là nguồn dữ liệu sẵn có, chi phí thấp hơn so với các hệ thống cảm biến chuyên dụng; do đó phù hợp cho các giải pháp giám sát thông minh.
- **Minh chứng nên lấy từ repo:** mục tiêu đề tài trong `task.md`, mô tả tổng quát dữ liệu trong `Dataset/README.md`, và cấu trúc pipeline trong `src/main.py`.
- **Gợi ý hình/bảng:** một hình mô tả bài toán tổng quát “SCADA -> tiền xử lý -> mô hình -> cảnh báo lỗi”.

### 2. Mục tiêu nghiên cứu

- **Cần viết:** xây dựng một quy trình hoàn chỉnh từ dữ liệu thô đến đánh giá mô hình phát hiện lỗi.
- **Mục tiêu cụ thể nên tách rõ:**
  1. tìm hiểu đặc trưng dữ liệu SCADA trong điện gió;
  2. thiết kế bước tiền xử lý và chuẩn hóa dữ liệu theo ranh giới asset/event;
  3. xây dựng và đánh giá sequence classifiers như LSTM, GRU, CNN-LSTM, CNN-GRU;
  4. khảo sát hướng sequence autoencoder cho cảnh báo bất thường.
- **Minh chứng nên lấy từ repo:** `src/data_pipeline/preprocessing/combined_sequence_pipeline.py`, `src/training/sequence_model_trainer.py`, `src/training/sequence_model_utils.py`.
- **Gợi ý bảng/hình:** bảng “mục tiêu - đầu ra - file mã nguồn liên quan”.

### 3. Đối tượng, phạm vi và ý nghĩa nghiên cứu

- **Cần viết:** đối tượng nghiên cứu là chuỗi dữ liệu SCADA 10 phút của các sự kiện bình thường và bất thường trong các wind farm.
- **Phạm vi triển khai thực tế của repo:** pipeline chính đang dùng cấu hình bài toán trong `configs/problem_v1.yaml` và `src/problem_config.py`, với Wind Farm A là phạm vi ban đầu.
- **Ý nghĩa khoa học:** kiểm chứng khả năng dùng mô hình học máy/học sâu để nhận diện trạng thái bất thường từ dữ liệu đa biến theo thời gian.
- **Ý nghĩa thực tiễn:** làm nền cho hệ thống cảnh báo sớm và bảo trì dự đoán trong môi trường vận hành thực tế.
- **Gợi ý hình/bảng:** bảng mô tả phạm vi “đã làm trong repo”, “đang thí nghiệm”, “đề xuất phát triển”.

### 4. Cấu trúc dự kiến của khóa luận

- **Cần viết:** giới thiệu ngắn gọn nội dung từng chương.
- **Có thể dùng chính README này** làm khung chuyển sang bản thảo khóa luận chính thức.

---

## Chương 1. Tổng quan nghiên cứu

### 1.1. Hệ thống SCADA trong giám sát tua-bin gió

- **Cần viết:** khái niệm SCADA, loại tín hiệu thu thập, tần suất lấy mẫu, vai trò của dữ liệu lịch sử trong giám sát tình trạng thiết bị.
- **Minh chứng nên lấy từ repo:** `Dataset/README.md` mô tả dữ liệu 95 event, 36 turbine, 3 wind farm, dữ liệu lấy mẫu mỗi 10 phút.
- **Điểm nên nêu:** dữ liệu gồm các nhóm nhiệt độ, tốc độ quay, điện, công suất gió, góc và trạng thái vận hành.
- **Gợi ý hình/bảng:** bảng phân loại nhóm cảm biến theo chức năng.

### 1.2. Bài toán phát hiện lỗi từ dữ liệu chuỗi thời gian đa biến

- **Cần viết:** phân biệt giữa anomaly detection, fault detection và fault prediction; giải thích vì sao dữ liệu SCADA là bài toán time-series đa biến, có nhiễu và mất cân bằng ngữ cảnh.
- **Nên liên hệ:** sự khác nhau giữa phát hiện ở mức timestep, mức cửa sổ thời gian và mức event.
- **Minh chứng nên lấy từ repo:** logic gom điểm theo `asset_id + sequence_id` trong `src/training/sequence_model_utils.py` và metadata cửa sổ trong `src/data_pipeline/preprocessing/combined_sequence_pipeline.py`.
- **Gợi ý hình/bảng:** sơ đồ chuyển đổi từ sample-level sang event-level decision.

### 1.3. Các hướng tiếp cận phổ biến trong nghiên cứu liên quan

- **Cần viết:** tóm tắt các nhóm phương pháp như thống kê/ngưỡng, cây quyết định, mạng hồi tiếp LSTM/GRU, autoencoder, CNN-LSTM và attention.
- **Cách trình bày nên dùng:** mỗi hướng tiếp cận mô tả ngắn ưu điểm, hạn chế, tính phù hợp với dữ liệu SCADA.
- **Minh chứng nên lấy từ repo:** sequence classifiers và autoencoder experiments trong `src/training/sequence_model_utils.py`, notebook và `autoencoder_results/`.
- **Gợi ý bảng/hình:** bảng so sánh “phương pháp - đầu vào - đầu ra - ưu điểm - hạn chế”.

### 1.4. Bài báo tham chiếu và vị trí của đề tài hiện tại

- **Cần viết:** giới thiệu bài báo tham chiếu như một hướng tiếp cận kết hợp CNN, LSTM và attention cho dữ liệu SCADA.
- **Phải viết trung thực:** repo hiện tại chưa triển khai đầy đủ mô hình CNN-LSTM-attention đó; thay vào đó, đề tài hiện tại đi theo hướng xây dựng pipeline combined-sequence vững chắc, ưu tiên mô hình phân loại chuỗi và autoencoder theo asset.
- **Cách diễn đạt nên dùng:** “bài báo được dùng làm tham chiếu cho cách tổ chức vấn đề và định hướng mở rộng, không phải là bản mô phỏng nguyên trạng”.
- **Gợi ý bảng/hình:** bảng đối chiếu “bài báo tham chiếu” và “repo hiện tại”.

### 1.5. Khoảng trống nghiên cứu và định hướng của khóa luận

- **Cần viết:** chỉ ra khoảng trống giữa lý thuyết và hiện thực triển khai, ví dụ: cần pipeline tái sử dụng được, giữ đúng ranh giới chuỗi, đánh giá nhất quán và khả năng mở rộng sang mô hình sâu hơn.
- **Minh chứng nên lấy từ repo:** sự tồn tại đồng thời của `src/`, `experiments/`, `report/`, `notebooks/` cho thấy dự án đang trong giai đoạn vừa phát triển hệ thống vừa khám phá thực nghiệm.
- **Gợi ý hình/bảng:** sơ đồ “khoảng trống nghiên cứu -> mục tiêu khóa luận -> đóng góp”.

---

## Chương 2. Cơ sở lý thuyết và dữ liệu

### 2.1. Mô tả bộ dữ liệu sử dụng

- **Cần viết:** bộ dữ liệu gồm 95 dataset, phân bố trên 36 turbine thuộc 3 wind farm A, B, C; trong đó 44 event bất thường và 51 event bình thường.
- **Nguồn mô tả nên dùng:** `Dataset/README.md` và `Dataset/raw/README.md`.
- **Điểm cần nhấn mạnh:** mỗi file CSV biểu diễn một event, có cả phần train và phần prediction; dữ liệu đã được ẩn danh nhưng vẫn bảo toàn cấu trúc phục vụ nghiên cứu.
- **Gợi ý bảng/hình:** bảng thống kê số event theo từng wind farm và số event theo nhãn anomaly/normal.

### 2.2. Cấu trúc dữ liệu và ý nghĩa các trường

- **Cần viết:** giải thích vai trò của `event_info.csv`, `feature_description.csv` và từng file event trong thư mục `datasets/`.
- **Các trường nên mô tả:** `event_id`, `event_label`, `event_start`, `event_end`, `time_stamp`, `asset_id`, `train_test`, `status_type_id` và các cột cảm biến `sensor_*`, `power_*`, `wind_speed_*`.
- **Minh chứng nên lấy từ repo:** `src/data_pipeline/loaders/event_loader.py`.
- **Gợi ý bảng/hình:** bảng “tên trường - kiểu dữ liệu - ý nghĩa - vai trò trong pipeline”.

### 2.3. Cơ sở lý thuyết về tiền xử lý dữ liệu SCADA

- **Cần viết:** làm sạch dữ liệu, điền khuyết bằng `ffill`, `bfill`, sau đó zero-fill; mã hóa đặc trưng góc bằng sin/cos để tránh đứt gãy chu kỳ; loại bỏ nhóm biến dạng counter nếu không phù hợp cho mô hình hóa trạng thái tức thời.
- **Minh chứng nên lấy từ repo:** `src/data_pipeline/preprocessing/feature_engineering.py` và các hằng số đặc trưng trong `src/config.py`.
- **Điểm cần phân tích:** lý do phải chuẩn hóa theo từng asset thay vì dùng một bộ chuẩn hóa chung.
- **Gợi ý hình/bảng:** sơ đồ luồng tiền xử lý từng event trước khi đưa vào mô hình.

### 2.4. Cơ sở lý thuyết về tạo chuỗi thời gian và chia tập

- **Cần viết:** cách chia dữ liệu train/validation/test theo thứ tự thời gian để tránh rò rỉ dữ liệu; vai trò của sliding window và stride trong biến đổi chuỗi.
- **Minh chứng nên lấy từ repo:** `src/data_pipeline/preprocessing/combined_sequence_pipeline.py`, `src/problem_config.py`, `configs/problem_v1.yaml`.
- **Thông số hiện có trong code nên nêu rõ:** dữ liệu có độ phân giải 10 phút; pipeline combined-sequence dùng cửa sổ 24 giờ (`W = 144` bước), horizon 12 giờ (`H = 72` bước) và `stride = 6`.
- **Lưu ý nên ghi vào khóa luận:** các cửa sổ không được vượt qua ranh giới `asset_id + sequence_id`, tránh rò rỉ dữ liệu giữa turbine, event và split.
- **Gợi ý bảng/hình:** hình minh họa cơ chế tạo cửa sổ và bảng ví dụ shape của `X_train`, `X_val`, `y_train`, `y_val`.

### 2.5. Cơ sở lý thuyết về các mô hình sử dụng

- **Sequence classifiers:** trình bày LSTM, GRU, CNN-LSTM và CNN-GRU như các mô hình phân loại chuỗi trực tiếp trên cửa sổ SCADA.
- **Sequence autoencoders:** trình bày LSTM Autoencoder và GRU Autoencoder như nhánh học hành vi bình thường theo từng asset, dùng reconstruction score để phát hiện bất thường.
- **Ngưỡng cảnh báo:** mô tả cách chọn threshold trên validation theo event-level F1 rồi báo cáo cả metric window-level và event-level.
- **Minh chứng nên lấy từ repo:** `src/training/sequence_model_trainer.py`, `src/training/sequence_model_utils.py`.
- **Gợi ý bảng/hình:** bảng so sánh input, output, mục tiêu học, và loại nhãn cần dùng cho từng mô hình.

---

## Chương 3. Phương pháp đề xuất và thiết kế hệ thống

### 3.1. Tổng quan pipeline của hệ thống

- **Cần viết:** pipeline thống nhất gồm hai giai đoạn `prepare -> train-sequences`, được điều phối từ `src/main.py`.
- **Mô tả nên theo luồng dữ liệu:** combined CSV -> kiểm tra schema -> nhóm theo `asset_id + sequence_id` -> chia train/val/test -> chuẩn hóa theo asset -> tạo cửa sổ -> huấn luyện mô hình chuỗi -> xuất kết quả.
- **Minh chứng nên lấy từ repo:** `src/main.py`.
- **Gợi ý hình/bảng:** sơ đồ kiến trúc tổng thể của hệ thống.

### 3.2. Chuẩn bị dữ liệu combined-sequence

- **Cần viết:** repo hiện tại ưu tiên combined-sequence pipeline, trong đó một CSV lớn giữ nguyên ranh giới `asset_id` và `sequence_id`.
- **Các bước nên mô tả:**
  1. đọc combined CSV và danh sách feature đã chọn;
  2. kiểm tra `time_stamp`, `asset_id`, `train_test`, `sequence_id`, `label/status_type_id`;
  3. chia train/validation/test bên trong từng `asset_id + sequence_id`;
  4. fit scaler trên train split;
  5. tạo cửa sổ theo window/horizon/stride;
  6. lưu export cho classifier và autoencoder trong `Dataset/processed/sequence_exports/window_<H>h/`.
- **Minh chứng nên lấy từ repo:** `src/data_pipeline/preprocessing/combined_sequence_pipeline.py`.
- **Gợi ý bảng/hình:** bảng cấu trúc thư mục `window_24h/classifier` và `window_24h/autoencoder`.

### 3.3. Mô hình phân loại chuỗi

- **Cần viết:** các mô hình LSTM, GRU, CNN-LSTM và CNN-GRU nhận trực tiếp cửa sổ SCADA và dự đoán xác suất có bất thường trong horizon tương lai.
- **Kiến trúc hiện có nên mô tả:** sequence encoder kết hợp lớp dense sigmoid để phân loại nhị phân ở mức cửa sổ.
- **Thông số mặc định trong code nên nêu:** `--classifier-epochs 25`, `--classifier-batch-size 256`, `--windows 24`, và class weight tính từ nhãn train.
- **Minh chứng nên lấy từ repo:** `src/training/sequence_model_trainer.py` và `src/training/sequence_model_utils.py`.
- **Gợi ý hình/bảng:** sơ đồ kiến trúc sequence classifier và bảng tham số huấn luyện.

### 3.4. Mô hình sequence autoencoder

- **Cần viết:** LSTM Autoencoder và GRU Autoencoder học tái tạo các cửa sổ bình thường theo từng asset; sai số tái tạo được dùng làm điểm bất thường.
- **Điểm nên phân tích:** autoencoder chỉ fit trên cửa sổ bình thường, chọn threshold bằng validation event-level F1, sau đó báo cáo event-level và window-level metrics.
- **Minh chứng nên lấy từ repo:** `src/training/sequence_model_trainer.py` và `src/training/sequence_model_utils.py`.
- **Gợi ý bảng/hình:** bảng so sánh classifier và autoencoder theo input, mục tiêu học và cách chọn threshold.

### 3.5. Thiết kế module phần mềm

- **Cần viết:** tách mô tả theo các nhóm module lớn.
- **Cấu trúc nên trình bày:**
  1. `data_pipeline`: nạp dữ liệu, tiền xử lý, tạo sequence, chuẩn hóa;
  2. `training`: huấn luyện classifier, autoencoder, threshold sweep và xuất metric;
  3. `inference`: tiện ích chạy mô hình đã lưu;
  4. `notebooks`, `reports` và `experiments`: phục vụ khảo sát, tổng hợp thực nghiệm và viết báo cáo.
- **Gợi ý bảng/hình:** bảng “module - chức năng - file tiêu biểu”.

---

## Chương 4. Thực nghiệm và đánh giá

### 4.1. Thiết lập thực nghiệm

- **Cần viết:** mô tả môi trường chạy, thư viện chính (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`), cách tổ chức thư mục dữ liệu và mô hình.
- **Minh chứng nên lấy từ repo:** `requirements.txt`, `configs/problem_v1.yaml`, `src/problem_config.py`.
- **Điểm nên nêu rõ:** pipeline chính sử dụng combined CSV cho bước `prepare` và sequence exports cho bước `train-sequences`; các wind farm còn lại có thể dùng cho khảo sát mở rộng hoặc thảo luận tổng quan dữ liệu.
- **Gợi ý bảng/hình:** bảng cấu hình môi trường và cấu hình thực nghiệm.

### 4.2. Thiết lập đánh giá và chỉ số đo lường

- **Cần viết:** Accuracy, Precision, Recall, F1-score, PR-AUC, ROC-AUC, lead time và false alarm ở cả window-level và event-level.
- **Minh chứng nên lấy từ repo:** `src/training/sequence_model_utils.py`, `src/training/sequence_model_trainer.py`.
- **Nên làm rõ:** đánh giá theo event giúp sát hơn với bài toán vận hành thực tế, nhưng window-level vẫn cần báo cáo để kiểm tra chất lượng cảnh báo cục bộ.
- **Gợi ý bảng/hình:** bảng định nghĩa chỉ số và ma trận nhầm lẫn mẫu.

### 4.3. Kịch bản thực nghiệm đề xuất

- **Kịch bản 1:** huấn luyện sequence classifiers gồm LSTM, GRU, CNN-LSTM và CNN-GRU trên export `window_24h`.
- **Kịch bản 2:** huấn luyện LSTM Autoencoder và GRU Autoencoder theo từng asset.
- **Kịch bản 3:** khảo sát ảnh hưởng của window size, scaler và tập feature đã chọn.
- **Kịch bản 4:** so sánh kết quả theo window-level và event-level để chọn mô hình báo cáo chính.
- **Gợi ý bảng/hình:** bảng “kịch bản - mô hình - dữ liệu - đầu ra mong đợi”.

### 4.4. Cách trình bày và phân tích kết quả

- **Cần viết:** trình bày bảng kết quả tổng hợp theo mô hình, sau đó phân tích sâu một số event điển hình.
- **Hướng phân tích nên có:**
  1. mô hình nào có recall tốt hơn;
  2. mô hình nào kiểm soát false alarm tốt hơn;
  3. khác biệt giữa classifier và autoencoder;
  4. lead time và false alarm có phù hợp với mục tiêu cảnh báo sớm không.
- **Minh chứng nên lấy từ repo:** `results/sequence_training_results/`, `reports/`, và các file metric do trainer xuất ra.
- **Gợi ý bảng/hình:** bảng tổng hợp metric, hình loss curve, score distribution, confusion matrix, case study timeline.

### 4.5. Thảo luận hạn chế của kết quả hiện tại

- **Cần viết:** phạm vi chính vẫn bắt đầu từ Wind Farm A; chưa có kiến trúc attention hoàn chỉnh trong `src/`; cần chuẩn hóa lại cấu hình window/horizon, môi trường Python và dữ liệu đầu vào trước khi chốt thực nghiệm cuối.
- **Có thể nêu thêm:** việc chọn window size, tiêu chí ngưỡng và cách chuẩn hóa cần được thống nhất khi chốt luận văn.
- **Gợi ý bảng/hình:** bảng “hạn chế - tác động - hướng khắc phục”.

---

## KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 5.1. Kết luận dự kiến

- **Cần viết:** khóa luận đã xây dựng được một pipeline phát hiện lỗi từ dữ liệu SCADA có thể tái sử dụng, kết nối từ dữ liệu thô đến đánh giá mô hình.
- **Điểm nên nhấn mạnh:** hướng combined-sequence giữ được ranh giới asset/event, hỗ trợ so sánh classifier và autoencoder, đồng thời tạo nền cho các nghiên cứu sâu hơn.

### 5.2. Hướng phát triển tiếp theo

- **Nên đề xuất:** triển khai đầy đủ CNN-LSTM với attention; khảo sát GRU, Transformer hoặc Temporal Convolutional Network; mở rộng đánh giá trên cả Wind Farm B/C; xây dựng cảnh báo thời gian thực; bổ sung giải thích mô hình và trực quan hóa nguyên nhân lỗi.
- **Cách liên hệ với bài báo tham chiếu:** xem kiến trúc attention như hướng nâng cấp tự nhiên của hệ thống hiện tại, không phải phần đã hoàn tất.
- **Gợi ý bảng/hình:** bảng roadmap nghiên cứu tiếp theo theo từng giai đoạn.

---

## TÀI LIỆU THAM KHẢO DỰ KIẾN

Khi đưa vào khóa luận chính thức, nên dùng kiểu tài liệu tham khảo đánh số `[1]`, `[2]`, `[3]` theo phong cách phổ biến trong báo cáo kỹ thuật tại Việt Nam. Có thể bắt đầu bằng khung sau:

```text
[1] L. Xiang, P. Wang, X. Yang, A. Hu, H. Su, "Fault detection of wind turbine based on SCADA data analysis using CNN and LSTM with attention mechanism", Measurement, vol. 175, 2021.
[2] Tài liệu mô tả bộ dữ liệu CARE to Compare và các nghiên cứu liên quan đến SCADA-based fault detection.
[3] Các bài báo tổng quan về condition monitoring, anomaly detection và predictive maintenance cho wind turbine.
[4] Tài liệu kỹ thuật hoặc báo cáo nội bộ của nhóm về mô hình LSTM, autoencoder và đánh giá thực nghiệm.
```

Nếu khoa yêu cầu chuẩn IEEE, Vancouver hoặc chuẩn nội bộ của khoa, có thể giữ thứ tự đánh số nhưng cần bổ sung đầy đủ DOI, số trang và thông tin hội nghị/tạp chí.

---

## PHỤ LỤC A. Gợi ý hình và bảng nên có trong khóa luận

- **Hình 1:** sơ đồ tổng quan bài toán phát hiện lỗi từ dữ liệu SCADA.
- **Hình 2:** kiến trúc tổng thể của pipeline `prepare -> train-sequences`.
- **Hình 3:** sơ đồ tạo sliding window theo `asset_id + sequence_id` và chuẩn hóa theo asset.
- **Hình 4:** kiến trúc sequence classifier.
- **Hình 5:** sơ đồ so sánh sequence classifier và sequence autoencoder.
- **Hình 6:** biểu đồ loss curve, confusion matrix, score distribution của mô hình tốt nhất.
- **Bảng 1:** mô tả bộ dữ liệu.
- **Bảng 2:** nhóm đặc trưng cảm biến.
- **Bảng 3:** siêu tham số mô hình.
- **Bảng 4:** kết quả thực nghiệm tổng hợp.
- **Bảng 5:** đối chiếu giữa bài báo tham chiếu và hệ thống hiện tại.

---

## PHỤ LỤC B. Phụ lục kỹ thuật bám theo codebase

### B.1. Cây thư mục rút gọn

```text
scada-fault-prediction/
|-- Dataset/
|   |-- raw/
|   |   |-- Wind Farm A/
|   |   |-- Wind Farm B/
|   |   `-- Wind Farm C/
|   `-- processed/
|-- src/
|   |-- data_pipeline/
|   |-- models/
|   |-- training/
|   |-- evaluation/
|   `-- main.py
|-- experiments/
|-- notebooks/
|-- autoencoder_results/
|-- report/
`-- README.md
```

### B.2. Các lệnh CLI chính theo `src/main.py`

```bash
# 1) Tạo sequence exports từ combined CSV
python src/main.py prepare --csv "D:/Data/df_final.csv" --feature-file "results/results/final_features.csv" --window-hours 24

# 2) Huấn luyện sequence classifiers và sequence autoencoders
python src/main.py train-sequences --windows 24

# 3) Tùy chọn: chỉ train classifier hoặc chỉ train autoencoder
python src/main.py train-sequences --windows 24 --skip-autoencoders
python src/main.py train-sequences --windows 24 --skip-classifiers
```

### B.3. Một số cờ hữu ích nên mô tả trong phụ lục báo cáo

- `--window-hours 24`: chọn trực tiếp kích thước cửa sổ cần export.
- `--window-candidates-hours 12 24 48`: thử nhiều kích thước cửa sổ trong bước search.
- `--combined-scaler minmax|standard`: chọn kiểu chuẩn hóa cho combined CSV.
- `--classifier-models lstm gru cnn_lstm cnn_gru`: chọn classifier cần train.
- `--autoencoder-models lstm_ae gru_ae`: chọn autoencoder cần train.
- `--assets 10 11`: giới hạn autoencoder theo một số asset cụ thể.

### B.4. Các file mã nguồn tiêu biểu nên trích dẫn trong phần phương pháp

- `src/main.py`: giao diện CLI thống nhất của hệ thống.
- `configs/problem_v1.yaml`: cấu hình bài toán, window, horizon, stride và metric.
- `src/problem_config.py`: loader/validator cho cấu hình bài toán.
- `src/data_pipeline/preprocessing/combined_sequence_pipeline.py`: pipeline tạo sequence exports.
- `src/training/sequence_model_trainer.py`: orchestration train classifier và autoencoder.
- `src/training/sequence_model_utils.py`: kiến trúc, metric, threshold sweep và xuất kết quả.

---

## Gợi ý sử dụng README này khi viết khóa luận

- Dùng nguyên cấu trúc chương mục bên trên để tạo đề cương báo cáo chính thức.
- Khi bắt đầu viết từng chương, giữ lại các dòng `Cần viết`, `Minh chứng nên lấy từ repo`, `Gợi ý hình/bảng` như checklist nội bộ; khi hoàn thiện bản khóa luận, có thể xóa các dòng checklist này để chuyển sang văn phong hoàn chỉnh.
- Nếu nhóm quyết định chuyển trọng tâm sang autoencoder hoặc CNN-LSTM, chỉ cần cập nhật lại Chương 3, Chương 4 và phần Kết luận, còn khung tổng quan và dữ liệu vẫn có thể giữ nguyên.
