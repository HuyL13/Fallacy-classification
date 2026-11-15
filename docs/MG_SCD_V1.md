# MG-SCD v1

Triển khai độc lập trong `src/mgscd`, theo ba tài liệu ở root:
`HUONG_DAN_TRIEN_KHAI_TIENG_VIET.md`, `MG_SCD_TECHNICAL_SPEC_V1.md`, `PROMPT_BOOK.md`.
Không thay đổi engine hoặc cấu hình Grounded v3. Chỉ classification 8 lớp trên
fallacious subset; `UNRESOLVED` là trạng thái, đầu ra forced-choice vẫn top-1.

## Chạy

Python >=3.11, từ root repository:

```bash
python -m pip install -r requirements-mgscd.txt -r requirements-dev.txt
python -m pytest -q
python -m src.mgscd.runner --config configs/classification.mgscd.mock.yaml --mock
```

Mock có một mẫu tổng hợp, sáu stage calls. Không phải kết quả thực nghiệm.
Runner từ chối output directory đã có dữ liệu để tránh trộn các phiên bản.
Đổi `output_dir` khi chạy lại.

Profile NVIDIA dùng đúng transport và endpoint hiện có trong repo. Export
`NVIDIA_API_KEY` trong shell; runner không tự đọc `.env`. Chạy dev nhỏ trước:

```bash
python -m src.mgscd.runner --config configs/classification.mgscd.v1.yaml --limit 5
```

`structured_output: false` phù hợp profile NVIDIA hiện tại: schema vẫn nằm trong
system prompt, trả về phải là JSON nghiêm ngặt. Chỉ bật native JSON Schema khi
endpoint/model hỗ trợ schema của stage (gồm dictionary điểm ordinal). Không có
fallback âm thầm khi provider từ chối schema. Model name/JSON support vẫn cần
xác minh bằng smoke API tại môi trường sử dụng; triển khai này chưa gọi API thật.

## Hợp đồng và kiểm chứng

1. Screening nhận đủ tám patterns, trả top-3 đến top-8 và ordinal scores giảm dần;
   hòa điểm thì thứ tự canonical ID. Gold không có trong bất kỳ prompt nào.
2. Mỗi builder chỉ nhận sample và pattern của mình, không nhận rank/score hay báo
   cáo bên kia. Chạy tuần tự để accounting/retry dễ kiểm tra, vẫn là hai request
   độc lập; không có khẳng định sai số độc lập thống kê.
3. Hai revision đều dựa trên hai assessment ban đầu, đúng một vòng. Không sửa
   criterion của đối thủ; không đếm quote cũ hoặc assessment y nguyên thành mới.
4. Một Arbiter nhận đủ hồ sơ. H2 được flip trực tiếp, không yêu cầu unanimity.
5. Verifier kiểm tra schema, criterion ID và quote nguyên văn trong target; không
   chứng minh quote hỗ trợ đúng ngữ nghĩa. Hai bên được dùng cùng quote.

`support/supported` ủng hộ nhãn; `countercheck/supported` chống nhãn. Unknown
được giữ riêng. Không tính điểm cơ chế bằng trọng số tùy tiện. Không có regex lock.

Screen lỗi: runner ghi failed sample, không sinh nhãn. Builder/revision/Arbiter
lỗi: giữ top-1 với route degraded và error. Adapter retry có giới hạn theo
`max_attempts` cho timeout/rate-limit/JSON/schema; lỗi 401 không retry.
Lỗi evidence/criterion sau khi parse được đánh dấu degraded ngay.

Target giữ nguyên Unicode, khoảng trắng, dấu câu; article lấy prefix 6000 ký tự,
log số ký tự gốc/đã dùng và trạng thái cắt. Context paper chỉ title/parent như
baseline; context article mới cung cấp plain-text article. Loader dùng official
splits, kiểm tra article/comment disjoint và lọc non-fallacious. Không tự tạo split.

## Pattern bank

`src/mgscd/patterns/eight_labels.v1.json` là operationalization có version/hash,
đóng băng định nghĩa Grounded v3 hiện có lúc triển khai. Form và cues được tham
khảo từ `COCOLOFA_PATTERNS`, không quyết định nhãn bằng cue. Đây không phải bản
trích nguyên văn paper hoặc bộ patterns đã được tác giả paper xác nhận.

Các lựa chọn cần giữ rõ trong báo cáo: authority/nature/tradition dùng warrant
quyết định hoặc không hợp lệ thay vì bắt buộc là lý do duy nhất; worse-problems
cho phép hai concern liên quan nhưng phân biệt được; slope yêu cầu escalation,
không đủ nếu chỉ dự đoán tiêu cực. Tiêu chí được viết mới từ các định nghĩa này.
Cần human audit trên dev trước khi chốt taxonomy/prompt cho thí nghiệm.

## Artifacts và đánh giá

Mỗi run ghi `manifest.json`, `pattern_bank.json`, `predictions.jsonl`, nếu có lỗi
screen thì thêm `failures.jsonl`; API thật thêm `raw_calls.jsonl`. Manifest giữ
hash source/prompt/pattern, cấu hình, versions, selected IDs, article groups,
kiểm tra split và snapshot hash các file baseline đang có. Run không resume tự
động; cache không được tái sử dụng giữa các điều kiện. Raw traces nằm trong
`outputs/` được gitignore; không ghi API key.

Mỗi API attempt ghi model/revision, stage/role, retry, raw request/response,
latency, response hash và token provider thực sự trả. Thiếu token/cost để null,
không giả định bằng zero. Cấu hình giá token để ước tính USD, không phải invoice.
Sáu stage calls khác số API attempts khi retry. Fast path một call.

`metrics.json` có accuracy, macro/per-class P/R/F1, top-2/top-3 recall,
beneficial/harmful flips, net correction, gate misses, degraded/failed/unresolved,
revision changes/new quotes/concessions và chi phí. Failed screens nằm trong N;
không có initial/final prediction nên không đóng góp flip. `debate_activation`
là route debate/N theo spec; `debate_attempt_rate` gồm cả degraded.

Đánh giá lại hoặc so sánh hai run cùng mẫu (paired article bootstrap):

```bash
python -m src.mgscd.offline --run outputs/RUN_A
python -m src.mgscd.offline --run outputs/RUN_A --compare-run outputs/RUN_B
```

Calibration là tùy chọn, không biến ordinal scores thành probabilities:

```bash
python -m pip install '.[calibration]'
python -m src.mgscd.offline --run outputs/DEV_RUN --fit-calibrator outputs/calibrator.json
python -m src.mgscd.runner --config configs/classification.mgscd.v1.yaml --calibrator outputs/calibrator.json
```

Lệnh cuối cần profile split test hoặc phần dev held-out khác ID, output directory
mới. Fitter chỉ chấp nhận split dev, cả dự đoán đúng và sai, coverage đầy đủ;
không cho dự đoán calibration trên ID đã fit. Đo Brier/ECE trên held-out;
logistic correctness không phải xác thực explanation.

## Ablations và giới hạn kết luận

Đặt `pipeline.variant` trong bản sao cấu hình:

| Điều kiện | Variant / cách chạy | Calls danh nghĩa |
|---|---|---|
| B0 | Single-agent baseline hiện có; cần dùng cùng frozen patterns khi so sánh | Theo baseline |
| B1 | Grounded v3 hiện có, giữ nguyên profile | 3 |
| B2 | `builders_aggregate`: so overall_support weak < mixed < strong, hòa giữ H1 | 3 |
| B3 | `no_revision`: builders + Arbiter, revision null | 4 |
| B4 | `no_counterchecks`: loại counterchecks ở mọi stage | 6 |
| B5 | `full`, `route_mode: debate_all` (mặc định) | 6 |
| B6 | `full`, `route_mode: margin_or_flags` sau tune dev | 1 hoặc 6 |
| B7 | Ensemble/self-consistency, cần matched actual token/cost | Theo budget |

B0/B1/B7 không được coi là đã chạy/đã matched budget bởi bảng này.
Ngưỡng 0.12 chỉ là placeholder. Không chạy/tune toàn test trước khi freeze.
Top-2 recall của MG-SCD chỉ có sau screening thật. Artifact Grounded v3 tìm thấy
trong workspace (`grounded-v3-smoke-20261111`) là offline mock ba mẫu, không thể
dùng làm baseline accuracy/coverage thực nghiệm. Các run legacy khác không thể
âm thầm thay cho Grounded v3 hiện tại.

Chưa có kết luận F1, superiority hoặc significance. Các test kiểm tra hợp đồng
và orchestration, không kiểm chứng năng lực model. Human semantic audit, real
provider smoke và controlled B0–B7 là bước thực nghiệm sau triển khai. Top-3
rescue, order-swap audit, vòng hai, retrieval và learned numeric judge là các
extension riêng, không nằm trong thuật toán MG-SCD v1 chính.
