# Phase 2 - Data Understanding cho CARE to Compare

Generated at: 2026-04-28 15:35:56

## Muc tieu

Phase 2 tap trung hieu du lieu truoc khi tien xu ly va huan luyen mo hinh. Ba wind farm A, B va C duoc doc theo cung mot quy trinh: kiem tra metadata, event labels, asset distribution, schema CSV, status distribution, train/prediction split, missing values va cac dac diem feature nhu angle/counter.

## Tong quan dataset

| farm | event_info_rows | event_csv_files | asset_count | anomaly_events | normal_events | csv_feature_columns_min | csv_feature_columns_max | angle_sensor_rows | counter_sensor_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wind Farm A | 22 | 22 | 5 | 12 | 10 | 81 | 81 | 4 | 0 |
| Wind Farm B | 15 | 15 | 9 | 6 | 9 | 252 | 252 | 3 | 4 |
| Wind Farm C | 58 | 58 | 22 | 27 | 31 | 952 | 952 | 12 | 0 |

## Chat luong du lieu da profile

Cac thong ke theo dong dang duoc tinh tren mau 2000 dong dau cua moi event CSV.

| farm | profiled_events | profiled_rows | mean_missing_feature_cell_pct | mean_expected_10min_gap_pct |
| --- | --- | --- | --- | --- |
| Wind Farm A | 22 | 44000 | 0.0 | 99.955 |
| Wind Farm B | 15 | 30000 | 0.0 | 99.977 |
| Wind Farm C | 58 | 116000 | 0.0 | 99.999 |

## Nhan xet nhanh cho bao cao

- Moi file trong `datasets/*.csv` nen duoc xem la mot event time series rieng, khong duoc tao window vuot qua ranh gioi event.
- `event_info.csv` la nguon nhan cap event chinh; `status_type_id` la trang thai van hanh theo timestamp va can loai khoi input feature neu no lam ro nhan.
- Ba wind farm co so chieu feature khac nhau rat lon, nen giai doan modeling nen danh gia theo tung farm truoc khi thu nghiem gop lien farm.
- `feature_description.csv` giup xac dinh angle/counter va la co so de mo ta y nghia feature trong capstone report.
- Khi sang Data Preparation, moi window can duoc group theo `farm + asset_id + event_id/sequence_id` de tranh leakage.

## Artifacts

Tables:
- `farm_overview`: `reports\phase2_data_understanding\tables\farm_overview.csv`
- `event_inventory`: `reports\phase2_data_understanding\tables\event_inventory.csv`
- `event_info_all`: `reports\phase2_data_understanding\tables\event_info_all.csv`
- `feature_description_all`: `reports\phase2_data_understanding\tables\feature_description_all.csv`
- `quality_summary`: `reports\phase2_data_understanding\tables\quality_summary.csv`
- `train_test_distribution`: `reports\phase2_data_understanding\tables\train_test_distribution.csv`
- `status_distribution`: `reports\phase2_data_understanding\tables\status_distribution.csv`
- `label_distribution`: `reports\phase2_data_understanding\tables\label_distribution.csv`
- `missing_by_feature_sample`: `reports\phase2_data_understanding\tables\missing_by_feature_sample.csv`

Figures:
- Số lượng event normal/anomaly theo wind farm: `reports\phase2_data_understanding\figures\event_label_counts_by_farm.png`
- Số lượng event theo từng turbine: `reports\phase2_data_understanding\figures\event_count_by_asset.png`
- Số lượng đặc trưng và nhóm đặc trưng đặc biệt: `reports\phase2_data_understanding\figures\feature_dimensions_by_farm.png`
- Phân bố train/prediction theo wind farm: `reports\phase2_data_understanding\figures\train_test_distribution.png`
- Phân bố trạng thái vận hành theo wind farm: `reports\phase2_data_understanding\figures\status_distribution.png`
- Phân bố thời lượng event theo wind farm: `reports\phase2_data_understanding\figures\event_duration_boxplot.png`