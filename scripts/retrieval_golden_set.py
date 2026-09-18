"""检索评测 golden set（docs/15 §7.7）。

每条：question / expected_sources（应命中的文件名，多个为空 OK）/ ground_truth（可选）。

构建规则：从 `data/` 真实内容出题，不合成。expected_sources 用文件 basename，
评测脚本会与召回结果的 metadata["source"] 比对（格式化为 basename 后匹配）。
"""

GOLDEN_SET = [
    # --- AcmeTech_Employee_Handbook.pdf ---
    {
        "question": "What are AcmeTech's regular office hours?",
        "expected_sources": ["AcmeTech_Employee_Handbook.pdf"],
        "ground_truth": "9:00 AM to 5:00 PM, Monday through Friday.",
    },
    {
        "question": "How many days of paid time off do employees get per year?",
        "expected_sources": ["AcmeTech_Employee_Handbook.pdf"],
        "ground_truth": "15 days per year, accrued monthly.",
    },
    {
        "question": "How many weeks of parental leave does AcmeTech provide?",
        "expected_sources": ["AcmeTech_Employee_Handbook.pdf"],
        "ground_truth": "12 weeks paid leave.",
    },
    {
        "question": "How many days per week may employees work remotely?",
        "expected_sources": ["AcmeTech_Employee_Handbook.pdf"],
        "ground_truth": "Up to three days per week, with manager approval.",
    },
    {
        "question": "What should an employee do upon receiving a suspicious email?",
        "expected_sources": ["AcmeTech_Employee_Handbook.pdf"],
        "ground_truth": "Report it to IT immediately.",
    },
    {
        "question": "What benefits are part of the AcmeTech financial package?",
        "expected_sources": ["AcmeTech_Employee_Handbook.pdf"],
        "ground_truth": "401(k) with company match, performance bonuses, and equity options.",
    },
    # --- AcmeTech_Product_Roadmap.md ---
    {
        "question": "What is the name of AcmeTech's flagship analytics product?",
        "expected_sources": ["AcmeTech_Product_Roadmap.md"],
        "ground_truth": "Atlas.",
    },
    {
        "question": "How many events per second can the Atlas stream processing handle?",
        "expected_sources": ["AcmeTech_Product_Roadmap.md"],
        "ground_truth": "Up to 50,000 events per second.",
    },
    {
        "question": "What are the three default RBAC roles in Atlas?",
        "expected_sources": ["AcmeTech_Product_Roadmap.md"],
        "ground_truth": "Viewer, Editor, Admin.",
    },
    {
        "question": "Which data sources does Atlas support natively?",
        "expected_sources": ["AcmeTech_Product_Roadmap.md"],
        "ground_truth": "PostgreSQL and MongoDB.",
    },
    {
        "question": "What is planned for the v2.5 release of Atlas?",
        "expected_sources": ["AcmeTech_Product_Roadmap.md"],
        "ground_truth": "Anomaly detection, custom alert thresholds, and data lineage view.",
    },
    {
        "question": "How can dashboards be delivered on a schedule?",
        "expected_sources": ["AcmeTech_Product_Roadmap.md"],
        "ground_truth": "Via email and Slack.",
    },
    # --- AcmeTech_Security_Policy.md ---
    {
        "question": "After how many inactive days are SSO accounts automatically disabled?",
        "expected_sources": ["AcmeTech_Security_Policy.md"],
        "ground_truth": "90 days.",
    },
    {
        "question": "What encryption is used for customer data at rest?",
        "expected_sources": ["AcmeTech_Security_Policy.md"],
        "ground_truth": "AES-256.",
    },
    {
        "question": "How often are encryption keys rotated?",
        "expected_sources": ["AcmeTech_Security_Policy.md"],
        "ground_truth": "Every 180 days.",
    },
    {
        "question": "How long do post-incident reviews have to produce a written report?",
        "expected_sources": ["AcmeTech_Security_Policy.md"],
        "ground_truth": "Within two weeks.",
    },
    {
        "question": "What is mandatory for all admin accounts?",
        "expected_sources": ["AcmeTech_Security_Policy.md"],
        "ground_truth": "Multi-factor authentication.",
    },
    {
        "question": "How often is access reviewed at AcmeTech?",
        "expected_sources": ["AcmeTech_Security_Policy.md"],
        "ground_truth": "Quarterly.",
    },
    # --- AcmeTech_Internal_Processes.md ---
    {
        "question": "What is the meal reimbursement limit for travelers?",
        "expected_sources": ["AcmeTech_Internal_Processes.md"],
        "ground_truth": "60 USD per day.",
    },
    {
        "question": "What purchase amount requires written manager approval?",
        "expected_sources": ["AcmeTech_Internal_Processes.md"],
        "ground_truth": "Above 5,000 USD.",
    },
    {
        "question": "When is a new-hire given production access?",
        "expected_sources": ["AcmeTech_Internal_Processes.md"],
        "ground_truth": "Only after the security review clears.",
    },
    {
        "question": "How many business days does it take to ship a new laptop?",
        "expected_sources": ["AcmeTech_Internal_Processes.md"],
        "ground_truth": "Three business days.",
    },
    {
        "question": "What happens to an employee's access on their last working day?",
        "expected_sources": ["AcmeTech_Internal_Processes.md"],
        "ground_truth": "Access is revoked before 5 PM.",
    },
    # --- AcmeTech_Support_FAQ.md ---
    {
        "question": "How many seats does the Starter plan include?",
        "expected_sources": ["AcmeTech_Support_FAQ.md"],
        "ground_truth": "Five seats.",
    },
    {
        "question": "What is the public API rate limit on paid plans?",
        "expected_sources": ["AcmeTech_Support_FAQ.md"],
        "ground_truth": "1,000 requests per minute.",
    },
    {
        "question": "What export formats are available for dashboards?",
        "expected_sources": ["AcmeTech_Support_FAQ.md"],
        "ground_truth": "PDF or CSV.",
    },
    {
        "question": "In how many days are pro-rated refunds available after renewal?",
        "expected_sources": ["AcmeTech_Support_FAQ.md"],
        "ground_truth": "14 days.",
    },
    {
        "question": "What usually causes empty reports?",
        "expected_sources": ["AcmeTech_Support_FAQ.md"],
        "ground_truth": "A timezone mismatch between workspace and schedule.",
    },
    {
        "question": "Where is customer data hosted?",
        "expected_sources": ["AcmeTech_Support_FAQ.md"],
        "ground_truth": "In the same region as the workspace.",
    },
]


class GoldenSet:
    def __init__(self) -> None:
        self.items = list(GOLDEN_SET)

    def __len__(self) -> int:
        return len(self.items)
