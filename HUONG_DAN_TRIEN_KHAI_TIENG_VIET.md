# HƯỚNG DẪN TRIỂN KHAI MG-SCD V1 (TIẾNG VIỆT)

**Mục tiêu:** chuyển ý tưởng MG-SCD thành pipeline có thể code, kiểm thử và phục vụ bài báo. Đây là đặc tả **đề xuất**, KHÔNG phải một phương pháp đã có kết quả thực nghiệm. Mã đi kèm đã được test bằng backend giả lập; **chưa được kiểm thử với API hoặc repository thực tế của bạn**.

## 1. Không nhầm với bản Grounded v3 hiện tại

Tài liệu `ARCHITECTURE.md` bạn cung cấp ghi chú ngay đầu rằng **Grounded v3 (11/11/2026)** dùng chung định nghĩa, phản biện theo claim, và **một lượt Arbiter** có quyền quyết định ngay cả khi Defender ủng hộ/chưa chốt alternative. Phần biểu đồ hai lượt hoán đổi thứ tự và yêu cầu đồng thuận 100% bên dưới thuộc kiến trúc **lịch sử**. Tài liệu cũng ghi Defender v1 đã chuyển regex/quote length thành cảnh báo, không còn dùng hard cue locking như mô tả cũ. Vì vậy MG-SCD không lấy quy tắc 'hai lượt đồng thuận mới được flip' làm baseline hiện tại.

Dự kiến tích hợp **bên cạnh** chứ không sửa trực tiếp cấu hình `configs/classification.adversarial.grounded-v3.nvidia.yaml`. Ta cần code thật của repo và `docs/GROUNDED_V3.md` trước khi làm adapter tương thích hoàn toàn.

## 2. Chốt phạm vi bài toán

V1 chỉ giải bài toán **classification 8 lớp** trên các mẫu đã được xác định là có fallacy. Đừng thêm `non_fallacious` vào 8 nhãn; nếu làm detection thì tạo experiment riêng. Tám ID chính xác xem `mgscd/schemas.py`. `UNRESOLVED` không phải nhãn thứ 9, chỉ là trạng thái nội bộ: khi bắt buộc một nhãn, hệ thống trả về top-1 ban đầu nhưng gắn `unresolved=true` và vẫn đánh giá lỗi bình thường.

Input chuẩn là `Sample(sample_id, target_comment, parent_comment, title, article, gold_label)`. `gold_label` chỉ đi vào hàm đánh giá offline, tuyệt đối không được đưa cho model trong bất kỳ prompt nào. Dataset loader phải bảo toàn original split của CoCoLoFa hoặc phân chia theo **article ID**; không để bình luận cùng bài báo rơi vào train/dev và test khác nhau.

## 3. Module 0 — Pattern Bank (mấu chốt học thuật)

Mỗi fallacy có các trường:

- `definition`: định nghĩa nghiệp vụ.
- `mechanism`: cơ chế lỗi suy luận (*premise -> conclusion thiếu cầu nối suy luận nào*).
- `logical_form`: biểu diễn logic khái quát.
- `cue_examples`: dấu hiệu tham khảo, không phải luật khóa.
- `criteria`: các mệnh đề cần kiểm tra, mỗi mệnh đề có ID, loại `support` hoặc `countercheck`, và câu mô tả.
- `contrasts`: nhãn thường dễ nhầm.
- `source_note`: nguồn, phiên bản, ghi chú định nghĩa.

Điểm phân biệt quan trọng:

- Với tiêu chí `support`: trạng thái `supported` **có lợi cho** nhãn đó.
- Với `countercheck`: trạng thái `supported` **bất lợi cho** nhãn đó.
- Trạng thái `refuted`: tiêu chí đó bị bác bỏ.
- Trạng thái `unknown`: chưa có đủ bằng chứng. Tuyệt đối **không đổi unknown thành refuted**.

Ví dụ cho *Slippery Slope*:

- `SS1/support`: có dự đoán chuỗi hệ quả từ quyết định ban đầu.
- `SS2/support`: chuỗi tiến tới kết quả nghiêm trọng hơn.
- `SS3/support`: bước chuyển nhân quả quan trọng thiếu cơ sở trong văn bản/ngữ cảnh được cung cấp.
- `SSX1/countercheck`: phát ngôn thực chất chỉ so sánh với vấn đề khác, nặng hơn, để bác bỏ vấn đề hiện tại.

Ví dụ cho *Appeal to Worse Problems*:

- `AW1/support`: có vấn đề B **khác** vấn đề đang thảo luận, được coi là nghiêm trọng hơn.
- `AW2/support`: mức độ nghiêm trọng của B được dùng để giảm nhẹ/gạt bỏ A.
- `AWX1/countercheck`: thực chất là dự đoán tương lai xấu đi từ chính A.

`patterns/eight_labels.example.json` đã có bản minh họa đủ 8 lớp. Tuy nhiên đó là **taxonomy do bản thiết kế này thao tác hóa**, không phải nguyên văn extracted patterns từ paper ArgMining 2026. Hãy thay bằng bộ enriched patterns bạn thực sự đang dùng và giữ version/hash. Nếu thay đổi ranh giới nhãn, phải khai báo rõ và làm ablation, không được coi là dùng nguyên xi định nghĩa cũ.

## 4. Module 1 — Screening & Uncertainty Estimator

**Mục đích:** lấy `top-3` để đo độ bao phủ nhãn; trong phiên bản chính, sử dụng `top-2` để tranh biện. Output gồm `ranked_labels`, `ordinal_scores`, `ambiguity_flags`, `notes`.

**Prompt:** `prompts/screen.txt`. Input: sample đã loại `gold_label` + **toàn bộ 8 patterns**.

**Thứ tự:** điểm trong `[0,1]`, giảm dần. Đây là điểm xếp hạng heuristic của LLM, **không được viết trong paper như xác suất đã hiệu chuẩn**. Ngưỡng `margin_threshold=0.12` chỉ là ví dụ, không có căn cứ để cố định nếu chưa thử trên dev.

**Chuyển nhánh:**

- Chạy `route_mode='debate_all'` trước để đo giá trị *debate thật*.
- Sau đó mới thử `route_mode='margin_or_flags'`. Debate khi `score1-score2 < threshold` **hoặc** có `ambiguity_flags`.
- Fast path: dùng label top-1; vẫn ghi log `route='fast'`.

**Vấn đề khoa học cần đo:** `top2_recall = # mẫu gold nằm trong top-2 / N`. Đây là trần khả năng sửa lỗi nếu Arbiter chỉ được chọn H1 hoặc H2. Nếu top-2 recall quá thấp, điều chỉnh screener hoặc so sánh top-3 trước khi tăng số rounds.

## 5. Module 2A/2B — Independent Candidate Builder & Challenger Builder

**Mục đích:** kiểm chứng riêng từng giả thuyết. H1 = top-1, H2 = top-2.

**Prompt chung:** `prompts/builder.txt` nhưng **hai request độc lập**:

- Builder H1 nhận sample và **chỉ** pattern H1.
- Builder H2 nhận sample và **chỉ** pattern H2.

Không được gửi `ordinal_scores`, kết luận của screener, tên/nhãn bên kia hoặc lập luận đã sinh bởi agent kia. Mục đích là giảm anchoring. Điều đó **không** bảo đảm thống kê hai sai số độc lập nếu dùng chung model.

**Output** `Hypothesis`:

- `label`: đúng nhãn được cấp.
- `premise`: tiền đề được xác định từ phát ngôn.
- `conclusion`: kết luận người nói muốn bảo vệ.
- `defect_mechanism`: diễn giải cụ thể lỗi suy luận nếu có.
- `criteria`: **đủ đúng từng criterion ID** của pattern, mỗi phần tử gồm ID, `status`, `quote`, `rationale`.
- `overall_support`: `strong | mixed | weak`.
- `strongest_quote`: quote mạnh nhất hoặc null.

**Lưu ý khắt khe:** chữ `supported` yêu cầu quote nguyên văn thuộc `target_comment`. Quote có thể trùng giữa hai agent và không bị khóa. Mệnh đề 'thiếu cầu nối suy luận' được giải thích bằng việc quote đoạn nhảy kết luận thiếu bằng chứng; không bịa ra những câu chưa từng xuất hiện. Nếu một tiêu chí không thể quyết định thì `unknown`.

**Vì sao Challenger không được ép phản đối?** Để tránh model bịa lý lẽ yếu chỉ vì role instruction. Builder H2 được quyền kết luận nhãn H2 rất yếu. Đây là sự khác biệt so với truyền thống Prosecutor/Defender buộc tranh tụng.

## 6. Module 3 — Selective Contrastive Debate (1 vòng)

Đây mới là phần tạo tương tác giữa agents. Sau hai builder, kiểm tra dữ liệu hợp lệ trước. Sau đó:

- Revision H1 nhận: own H1 assessment + opponent H2 assessment + hai patterns + sample.
- Revision H2 nhận: own H2 assessment + opponent H1 assessment + hai patterns + sample.

**Prompt:** `prompts/revision.txt`.

Mỗi bên phải:

1. Kiểm tra đối thủ dựa trên **criterion ID cụ thể**, không phản đối chung chung.
2. Tự rà soát xem tiêu chí nào của mình cần **sửa** dựa trên bằng chứng mới.
3. Có thể nhận nhãn bên kia tốt hơn: `concedes_opponent_better=true`.
4. Chỉ trả về `revisions` gồm các criterion thực sự thay đổi; nếu không phát hiện mới, trả `[]`.
5. `new_target_quotes` chỉ được chứa chuỗi nguyên văn trong phát ngôn mục tiêu.

Một vòng là đủ cho V1. Trước khi thêm vòng 2, phải định nghĩa được tiêu chí 'new information' và thực nghiệm chứng minh có lợi. Dùng 2 rounds ngay từ đầu sẽ khó chứng minh thành phần nào thực sự cải thiện.

**Lưu ý cho paper:** một câu phản bác mới trong lời văn không đồng nghĩa có thông tin mới. Hãy đo tỉ lệ criterion thay đổi trạng thái có xác thực, số quote mới hợp lệ, tỷ lệ nhượng bộ và eventual beneficial flips.

## 7. Module 4 — Evidence-Conditioned Arbiter

**Prompt:** `prompts/arbiter.txt`. Nhận cả 2 đánh giá ban đầu, 2 revision, 2 pattern và sample. Không biết gold.

**Cách quyết định:** so sánh nhãn theo cơ chế lỗi và các điều kiện phân biệt:

- `supported` trên support criterion ủng hộ nhãn.
- `supported` trên countercheck criterion chống lại nhãn.
- `unknown` KHÔNG tự suy ra nhãn kia đúng.
- Lời tự tin/nhượng bộ của agent chỉ là gợi ý, không đủ làm chứng cứ.
- Quote bề mặt không đủ; cần `premise -> conclusion -> reasoning defect`.

**Output:** `decision='H1'|'H2'|'UNRESOLVED'`, `decisive_criteria` hợp lệ, `justification`, `needs_more_context`.

**Logic cuối:** H1 => giữ ban đầu; H2 => flip; UNRESOLVED => giữ top-1 nhưng đánh dấu. V1 không áp dụng unanimous flip của kiến trúc legacy. Có thể chạy **order-swap audit trên dev** để đo vị trí ảnh hưởng tới Arbiter, không âm thầm biến audit thành thuật toán chính.

**Chưa có công thức tối ưu**: nếu muốn cơ chế số `S(H)=w_eE+w_mM-w_vV`, cần định nghĩa E/M/V, chọn trọng số trên dev và so sánh với Arbiter LLM. V1 sử dụng Arbiter cấu trúc để tránh phải lựa chọn trọng số tùy tiện.

## 8. Module 5 — Verifier và Calibrator

Verifier là code Python, không dùng thêm một LLM đắt tiền. `mgscd/validators.py` kiểm tra:

- JSON đúng schema và nhãn nằm trong tám loại.
- IDs của pattern đủ, không lặp, không bịa.
- Quote là chuỗi con nguyên văn của target.
- `supported` phải có quote.
- Revision không được chỉnh tiêu chí thuộc pattern của đối thủ.
- Arbiter không được tham chiếu criterion không tồn tại.

Verifier **không chứng minh tính đúng đắn ngữ nghĩa** của một quote. Bạn vẫn phải manual audit trên dev mẫu. Đặc biệt cần kiểm tra quote đúng chữ nhưng cơ chế kết luận sai.

**Confidence:** `ordinal_scores` ban đầu chưa có căn cứ là xác suất. Có file `mgscd/calibration.py` cung cấp một post-hoc calibrator được huấn luyện trên dev; chỉ sau đó mới thêm `calibrated_confidence`. Cần báo cáo Brier/ECE, vì calibrator vẫn có thể sai.

## 9. Fallback và lỗi không được giấu

- **Screen lỗi:** exception, không tự sinh nhãn. Xử lý retry hoặc failed samples minh bạch.
- **Builder/Revision/Arbiter lỗi:** giữ top-1, `route='degraded'`, `degraded=true`, ghi `error`.
- **UNRESOLVED:** vẫn được tính bình thường trong forced 8-way accuracy; thêm `unresolved=true`.
- **API timeout/rate limit:** retry có giới hạn cố định, lưu token/latency; nếu hết retry phải ghi failed/degraded.
- **No article/parent:** cho phép chuỗi rỗng, không giả định ngữ cảnh có thật.
- **Long article:** giữ nguyên target; cắt article theo chính sách nhất quán và ghi việc cắt.
- **Ambiguous semantics:** không biến quote check thành semantic truth.

## 10. Các trường hợp đặc biệt bắt buộc test

1. Hai agent cùng trích dẫn nguyên văn một câu giống nhau — **hợp lệ**.
2. H1 đúng và H2 cố bịa sự kiện ngoài bài — H2 phải yếu, không ép tranh cãi.
3. H1 sai, H2 đúng, H2 có bằng chứng thực — Arbiter phải có khả năng flip.
4. Top-2 không chứa nhãn đúng — ghi rõ giới hạn, không dùng oracle gold ở inference.
5. Tiêu chí `countercheck` được support — phải là bằng chứng **chống** nhãn đó.
6. Dữ liệu có một câu trích giống nội dung bài báo nhưng không thuộc target — quote phải bị bác bỏ nếu bị gắn source=target.
7. JSON sai, thiếu field, trùng criterion, nhãn ngoài tám loại — phải lỗi rõ ràng.
8. `UNRESOLVED` không được tính thành nhãn riêng; fallback có đánh dấu.
9. Early exit sai nhưng rất tự tin — đo tỉ lệ bỏ sót và đánh giá threshold trên dev.
10. Có instruction injection trong target ('Ignore earlier instructions...') — dữ liệu phải chỉ được xem là text để phân loại.
11. Unicode, dấu câu, quote lặp trong target, text rất ngắn, article rất dài.
12. Kết quả MỘT lần chạy mock không phải bằng chứng cải thiện F1.

## 11. Cách chạy code mẫu

```bash
cd mgscd_v1
python -m pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python examples/run_mock.py
```

Trong `examples/mock_backend.py`, câu trả lời của model là dữ liệu giả lập. Nó chỉ kiểm chứng code orchestration. Sau khi các test qua, cần viết adapter theo `JSONLLM.complete(stage, system, user, sample_id)`. Có `examples/openai_compatible_backend.py` dùng SDK `openai` cho server tương thích giao thức chat-completions; **chưa được xác minh với endpoint NVIDIA hoặc model cụ thể của bạn**. Đọc kỹ API hỗ trợ JSON mode/temperature/token limit trước khi chạy.

Ví dụ nếu server hỗ trợ đúng OpenAI-style request:

```bash
python -m pip install openai
export MG_SCD_BASE_URL='https://YOUR_PROVIDER_API_BASE/v1'
export MG_SCD_API_KEY='YOUR_SECRET_IN_ENV_ONLY'
export MG_SCD_MODEL='YOUR_EXACT_MODEL_ID'
PYTHONPATH=. python examples/run_real_template.py
```

Không lưu API key vào file nguồn, log hay paper. Đừng chạy nguyên test set trước khi prompt và threshold được chốt.

## 12. Số lần gọi, chi phí và khả năng song song

- Fast path: **1 call** screening.
- Debate path: **1 screening + 2 builder + 2 revision + 1 arbiter = 6 calls**.
- Builder H1/H2 có thể chạy song song; revision H1/H2 có thể song song sau khi hai builder hoàn thành. Arbiter luôn đợi cả hai.
- Verifier là code nên **0 call**. Calibrator hậu xử lý cũng **0 call**.
- Đây là ngân sách *không tính retries*. Khi có lỗi phải báo cả calls thực tế lẫn tokens và cost.
- `route_mode='debate_all'` vẫn **không** phải debate tự do nhiều rounds: nó chỉ có một vòng phản biện có cấu trúc.

## 13. Ablation để chứng minh đóng góp

Chạy trên cùng mẫu, cùng version model/pattern/context/prompt budget càng gần nhau càng tốt:

- **B0:** single-agent với patterns hiện tại.
- **B1:** Grounded v3 hiện tại, không sửa mã gốc.
- **B2:** 2 builders độc lập, aggregate đơn giản (không revision).
- **B3:** 2 builders + arbiter (không revision).
- **B4:** 2 builders + revision + arbiter **không counterchecks**.
- **B5:** full MG-SCD nhưng debate tất cả.
- **B6:** full MG-SCD chỉ debate khi ambiguous (sau tune dev).
- **B7:** self-consistency / ensemble với **cùng token/cost**.

Chỉ số bắt buộc: accuracy, macro-F1, per-class F1, top2 recall, beneficial flips, harmful flips, net correction, debate activation, degraded rate, token/latency/cost. `net_correction=(#sai->đúng)-(#đúng->sai)`; chia N sẽ bằng delta accuracy nếu cả hai đều là forced predictions 8 lớp trên cùng N mẫu. Phân tích ý nghĩa thống kê bằng paired bootstrap, tốt nhất cluster theo bài báo.

## 14. Tiêu chuẩn dừng/kết luận

Nếu builder đối chiếu + revision **không tăng net correction**, hoặc chỉ cải thiện nhờ có thêm budget, chưa đủ cơ sở để báo cáo innovation về debate. Khi đó thử đóng góp nhỏ hơn như taxonomy-boundary checker, top-k ranking, hoặc nghiên cứu giới hạn của debate. Nếu B5 vượt B3/B4 và B7 ở mức ý nghĩa phù hợp, đó là bằng chứng hỗ trợ claim. Đừng viết 'SOTA' trước khi kiểm tra baseline fine-tuned encoder trên CoCoLoFa và thực nghiệm kiểm soát đầy đủ.

## 15. Cơ sở tài liệu

1. Papadopulos et al. (ArgMining 2026), *Beyond Logical Forms*: https://aclanthology.org/2026.argmining-1.2/ — cơ sở cho enriched patterns.
2. Yeh et al. (EMNLP 2024), *CoCoLoFa*: https://aclanthology.org/2024.emnlp-main.39/ — dữ liệu và baseline.
3. Smit et al. (ICML 2024), *Should we be going MAD?*: https://proceedings.mlr.press/v235/smit24a.html — tranh biện không luôn tốt hơn ensemble.
4. Ku et al. (ArgMining 2025), *Multi-Agent LLM Debate Unveils the Premise Left Unsaid*: https://aclanthology.org/2025.argmining-1.6/ — phản biện để sửa giả thuyết, không gượng ép bảo vệ role.
5. Calvo Figueras et al. (ArgMining 2025), *Critical Questions Generation*: https://aclanthology.org/2025.argmining-1.23/ — cơ sở cho kiểm chứng có cấu trúc.
6. Wang et al. (ACL 2024), *Large Language Models are not Fair Evaluators*: https://aclanthology.org/2024.acl-long.511/ — căn cứ kiểm tra order bias.

**Nguồn giấy tờ cung cấp cơ sở cho từng thành phần, không chứng minh trước MG-SCD có hiệu quả.**
