"""
Augment the prompt profiling dataset with targeted improvements:
  1. +45 vague T2 prompts (from 17 → 62)
  2. +20 D5=1.0 T3 prompts (add missing class)
  3. +15 boundary hard-T2 prompts (cs 0.60–0.69)
  4. +15 boundary soft-T3 prompts (cs 0.70–0.80)
  5. +15 T3 with explicit D4 textual signals
  6. +20 T1 balance prompts
  7. +15 T2 explicit/implicit variety

Saves: dataset_prompt_profiling_v2.csv
"""
import pandas as pd

# ── Load original dataset ────────────────────────────────────────────────────
df_orig = pd.read_csv('dataset_prompt_profiling.csv')
print(f"Original dataset: {len(df_orig)} rows")

# ── New prompts ──────────────────────────────────────────────────────────────
# Each tuple: (id, prompt, phrasing_style, domain, d1, d2, d3, d4, d5)
# Complexity formula: cs = d1*0.35 + d2*0.20 + d3*0.20 + d4*0.15 + d5*0.10
# Tiers: T1 < 0.40 | T2 [0.40, 0.70) | T3 >= 0.70

new_rows = [
    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH 1: VAGUE T2 (45 prompts) — medium complexity, unclear phrasing
    # ═══════════════════════════════════════════════════════════════════════════

    # IoT & Smart Factory
    ("v3_vt2_001", "factory sensors keep disconnecting every few hours and nobody can figure out why, production is slowing down", "vague", "IoT & Smart Factory", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125
    ("v3_vt2_002", "plc firmware updates are breaking our assembly line scripts and the integrator team is blaming our network", "vague", "IoT & Smart Factory", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875+0.025=0.5125... let me recalculate
    # d1=0.75*0.35=0.2625, d2=0.50*0.20=0.10, d3=0.50*0.20=0.10, d4=0, d5=0.25*0.10=0.025 → cs=0.4875 → T2 ✓
    ("v3_vt2_003", "our edge gateway is maxing out cpu and we think its the mqtt broker but not sure", "vague", "IoT & Smart Factory", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_004", "scada dashboard is showing stale readings for half the floor and the vendor says its our firewall rules", "vague", "IoT & Smart Factory", 0.75, 0.50, 0.50, 0.00, 0.00),  # cs=0.4625 → T2 ✓

    # Marketing Tech
    ("v3_vt2_005", "email deliverability tanked after we migrated to the new esp and open rates are in the gutter", "vague", "Marketing Tech", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_006", "our attribution numbers dont match between google analytics and the crm and marketing cant figure out their real roas", "vague", "Marketing Tech", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_vt2_007", "lead scoring in hubspot is all over the place and sales keeps getting garbage leads from the automated workflows", "vague", "Marketing Tech", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_008", "utm tracking is completely broken after the website redesign and we have no idea which campaigns are actually working", "vague", "Marketing Tech", 0.75, 0.25, 0.75, 0.00, 0.25),  # cs=0.4875 → T2 ✓

    # Supply Chain
    ("v3_vt2_009", "warehouse picking errors spiked this month and the wms team says its a slotting problem but ops disagrees", "vague", "Supply Chain", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_010", "our demand forecasting model keeps overestimating for seasonal products and inventory is piling up in three warehouses", "vague", "Supply Chain", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_vt2_011", "purchase orders are getting stuck between sap and the supplier portal and nobody knows which system is dropping them", "vague", "Supply Chain", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_012", "freight invoices keep coming in wrong and our ap team is manually fixing hundreds of line items every week", "vague", "Supply Chain", 0.75, 0.25, 0.50, 0.00, 0.25),  # cs=0.4375 → T2 ✓

    # DevOps
    ("v3_vt2_013", "container images are bloated and our registry is eating storage like crazy plus deploys are slow", "vague", "DevOps", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_014", "test coverage keeps dropping every sprint and nobody owns the test infrastructure anymore", "vague", "DevOps", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_015", "our monitoring stack is generating so many false positive alerts that the on-call team ignores everything now", "vague", "DevOps", 0.75, 0.50, 0.50, 0.00, 0.00),  # cs=0.4625 → T2 ✓
    ("v3_vt2_016", "dev environments are drifting from prod and we keep finding bugs that only show up after deploy", "vague", "DevOps", 0.75, 0.25, 0.50, 0.00, 0.25),  # cs=0.4375 → T2 ✓

    # Cloud Infrastructure
    ("v3_vt2_017", "lambda cold starts are killing our api response times and the frontend team is complaining about timeouts", "vague", "Cloud Infrastructure", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_018", "our vpc peering setup is a mess and traffic between services is hitting the public internet instead of staying internal", "vague", "Cloud Infrastructure", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_vt2_019", "rds connection pool is maxing out during peak hours and the app just crashes", "vague", "Cloud Infrastructure", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_020", "our multi-az setup doesnt seem to be working because the failover test took way too long last week", "vague", "Cloud Infrastructure", 0.75, 0.50, 0.50, 0.00, 0.00),  # cs=0.4625 → T2 ✓

    # FinOps
    ("v3_vt2_021", "data transfer costs between regions are insane and we dont even know which services are causing it", "vague", "FinOps", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_022", "reserved instances are expiring next month and nobody has a plan for whether to renew or switch to savings plans", "vague", "FinOps", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_vt2_023", "dev team keeps spinning up expensive gpu instances for testing and forgetting to shut them down", "vague", "FinOps", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_024", "our cloud tagging strategy is a disaster and cost allocation reports are useless for the finance team", "vague", "FinOps", 0.75, 0.25, 0.75, 0.00, 0.25),  # cs=0.4875 → T2 ✓

    # Security
    ("v3_vt2_025", "access reviews are overdue and we have no idea who has admin access to what anymore", "vague", "Security", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_026", "secrets are scattered across environment variables and config files and nobody has a central vault setup", "vague", "Security", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_vt2_027", "our ssl certificates keep expiring without warning and production goes down every few months because of it", "vague", "Security", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_028", "phishing emails are getting past our email gateway and employees keep clicking on them", "vague", "Security", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓

    # Data Engineering
    ("v3_vt2_029", "our dbt models are taking forever to run and the warehouse costs are spiking every morning", "vague", "Data Engineering", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_030", "data quality is terrible downstream because upstream sources keep changing schemas without telling anyone", "vague", "Data Engineering", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_vt2_031", "spark jobs are failing randomly with out of memory errors and nobody wants to tune the cluster config", "vague", "Data Engineering", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_032", "our etl pipeline missed its sla three times this week and the bi team is showing stale reports to executives", "vague", "Data Engineering", 0.75, 0.50, 0.50, 0.00, 0.00),  # cs=0.4625 → T2 ✓

    # AI Governance
    ("v3_vt2_033", "devs are using chatgpt to write code and pasting proprietary source code into it without any policy in place", "vague", "AI Governance", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_034", "our recommendation engine is showing biased results and the product team doesnt know how to fix it", "vague", "AI Governance", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓

    # System Integration
    ("v3_vt2_035", "salesforce and our erp are out of sync again and orders are falling through the cracks", "vague", "System Integration", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_036", "api rate limits from our payment provider keep throttling our checkout flow during peak hours", "vague", "System Integration", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓

    # HR Tech
    ("v3_vt2_037", "onboarding workflow in workday is broken and new hires are waiting days to get their accounts provisioned", "vague", "HR Tech", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_vt2_038", "performance review forms are a mess and managers are skipping calibration steps because the tool is confusing", "vague", "HR Tech", 0.75, 0.25, 0.50, 0.00, 0.25),  # cs=0.4375 → T2 ✓

    # General Enterprise
    ("v3_vt2_039", "our internal wiki is completely outdated and teams are making decisions based on stale documentation", "vague", "General Enterprise", 0.75, 0.25, 0.50, 0.00, 0.00),  # cs=0.4125 → T2 ✓
    ("v3_vt2_040", "meeting room booking system is double-booking rooms and people keep showing up to occupied rooms", "vague", "General Enterprise", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓

    # Regulatory Compliance
    ("v3_vt2_041", "gdpr data deletion requests are piling up and we have no automated process to handle them", "vague", "Regulatory Compliance", 0.75, 0.25, 0.50, 0.00, 0.25),  # cs=0.4375 → T2 ✓
    ("v3_vt2_042", "our consent management platform is not syncing with the marketing database and we might be emailing opted-out users", "vague", "Regulatory Compliance", 0.75, 0.50, 0.50, 0.00, 0.00),  # cs=0.4625 → T2 ✓

    # Competitive Intelligence
    ("v3_vt2_043", "our internal reporting dashboards are showing completely different numbers than what finance is presenting to the board", "vague", "General Enterprise", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_vt2_044", "customer churn rate jumped and nobody has analyzed why or what changed in the product", "vague", "Marketing Tech", 0.75, 0.25, 0.75, 0.00, 0.00),  # cs=0.4625 → T2 ✓
    ("v3_vt2_045", "our mobile app crash rate doubled after the last release and support tickets are flooding in", "vague", "DevOps", 0.75, 0.50, 0.50, 0.00, 0.00),  # cs=0.4625 → T2 ✓

    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH 2: D5=1.0 T3 (20 prompts) — multi-deliverable complex output
    # ═══════════════════════════════════════════════════════════════════════════

    ("v3_d5hi_001", "I have uploaded our current cloud architecture diagrams and vendor contracts. Build a comprehensive multi-cloud cost optimization proposal that includes an executive summary report, detailed cost modeling spreadsheets, a 3-year savings roadmap with milestones, a vendor negotiation playbook, and a governance policy document for ongoing cost control, all benchmarked against current market pricing from AWS, Azure and GCP", "explicit", "FinOps", 1.00, 1.00, 1.00, 1.00, 1.00),  # cs=1.0 → T3 ✓
    ("v3_d5hi_002", "Using our attached compliance audit findings and system architecture documentation, produce a full NIST 800-53 compliance remediation package including a gap analysis report, a prioritized remediation roadmap, updated security policy documents, technical runbooks for each control family, and a board-ready executive risk summary with current industry benchmark comparisons", "explicit", "Security", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.35+0.20+0.20+0.1125+0.10=0.9625 → T3 ✓
    ("v3_d5hi_003", "Our data platform needs a complete overhaul. Based on the attached current architecture and data catalog, deliver an end-to-end data platform modernization blueprint that includes architecture diagrams, a migration runbook, data quality framework documentation, cost projections, and a phased implementation timeline with go/no-go criteria for each phase, comparing Snowflake vs Databricks vs BigQuery pricing", "explicit", "Data Engineering", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_004", "Using the attached HR system configurations and employee data schemas, create a global workforce analytics platform specification that includes system architecture documentation, integration specifications for Workday and SAP SuccessFactors, a data governance policy, dashboard wireframes with KPI definitions, and a regulatory compliance matrix covering GDPR and local labor laws across our 12 operating countries, referencing current vendor pricing models", "explicit", "HR Tech", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_005", "Based on the uploaded supply chain data and logistics contracts, develop a complete supply chain resilience program that includes a risk assessment report, alternative supplier evaluation matrix with current market pricing, logistics optimization models, a crisis response playbook, and a quarterly monitoring dashboard specification", "explicit", "Supply Chain", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_006", "We need a full enterprise AI governance framework. Deliver a comprehensive package that includes a corporate AI usage policy document, a technical risk assessment methodology, model evaluation checklists, a compliance mapping to EU AI Act and NIST AI RMF, incident response procedures for AI failures, and a training curriculum outline for engineering teams, referencing current industry best practices and competitor approaches", "explicit", "AI Governance", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_007", "Using the attached marketing analytics exports and campaign performance data, build a complete marketing technology consolidation proposal including a current state assessment report, a vendor comparison matrix with live pricing from Segment, Tealium and mParticle, an integration architecture blueprint, a data migration playbook, and an ROI projection model for the executive team", "explicit", "Marketing Tech", 1.00, 1.00, 1.00, 1.00, 1.00),  # cs=1.0 → T3 ✓
    ("v3_d5hi_008", "Create a comprehensive DevSecOps transformation program for our engineering organization including a maturity assessment report, a toolchain architecture with CI/CD pipeline specifications, security scanning integration guides, a developer training program outline, a compliance evidence collection framework for SOC 2, and a phased rollout timeline with success metrics", "explicit", "DevOps", 1.00, 1.00, 0.75, 0.50, 1.00),  # cs=0.35+0.20+0.15+0.075+0.10=0.875 → T3 ✓
    ("v3_d5hi_009", "Develop a full IoT platform modernization strategy including current state architecture documentation, a sensor integration specification covering MQTT and OPC-UA protocols, edge computing deployment guides, a data pipeline architecture from factory floor to cloud analytics, a security hardening playbook, and a cost-benefit analysis comparing major IoT platform vendors at current market rates", "explicit", "IoT & Smart Factory", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_010", "Based on the attached system integration inventory, deliver a complete enterprise integration platform proposal including an API gateway architecture design, message queue topology diagrams, a data transformation mapping document, error handling and retry specifications, a vendor evaluation comparing MuleSoft vs Boomi vs Azure Integration Services with current pricing, and a migration runbook from our legacy middleware", "explicit", "System Integration", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_011", "Our CISO wants a comprehensive cloud security posture assessment delivered as a complete package including a vulnerability assessment report, a compliance gap matrix against ISO 27001 and SOC 2, a cloud security architecture blueprint, hardening runbooks for AWS and Azure, an incident response playbook, and an executive risk dashboard specification with industry benchmarking data", "implicit", "Security", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_012", "The board wants a full digital transformation assessment. We need a current state maturity report, a target state architecture vision, a technology investment roadmap with vendor pricing analysis, a change management program outline, a governance framework document, and a financial model projecting ROI over five years across our AWS and Azure infrastructure", "implicit", "Cloud Infrastructure", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_013", "We are building an enterprise data mesh and need the complete deliverable set: a data domain ownership model, a self-serve data platform architecture, data product specifications, a federated governance policy, implementation guides for each domain team, and a cost projection model comparing implementation approaches using current Snowflake and Databricks pricing tiers", "implicit", "Data Engineering", 1.00, 1.00, 0.75, 0.75, 1.00),  # cs=0.35+0.20+0.15+0.1125+0.10=0.9125 → T3 ✓
    ("v3_d5hi_014", "Our legal team needs a complete AI vendor risk assessment package including due diligence questionnaires for each of our five LLM providers, a comparative analysis of their data processing agreements against GDPR requirements, technical security assessment reports, a risk scoring matrix, remediation recommendations, and a contract amendment playbook referencing current market terms and competitor vendor offerings", "implicit", "AI Governance", 1.00, 1.00, 1.00, 1.00, 1.00),  # cs=1.0 → T3 ✓
    ("v3_d5hi_015", "Build a complete global payroll system migration package for our expansion into 8 new countries including a regulatory compliance matrix for each jurisdiction, a system architecture design for Workday integration, data migration specifications, a testing and validation playbook, a parallel-run monitoring plan, and localized tax configuration guides referencing current tax authority requirements in each country", "implicit", "HR Tech", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_016", "We need to deliver a comprehensive FinOps maturity program to the CFO including a current state assessment with cloud spend benchmarking against industry peers, a showback and chargeback policy document, an optimization playbook covering reserved instances and savings plans across three clouds, an anomaly detection alerting specification, a FinOps team charter, and a quarterly business review template with KPIs", "implicit", "FinOps", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_017", "Create a full marketing attribution overhaul package including a current state audit report, a server-side tracking implementation guide for Google and Meta conversion APIs, a first-party data strategy document, a consent management integration specification, a dashboard redesign with multi-touch attribution models, and an executive presentation comparing our performance metrics against industry benchmarks", "explicit", "Marketing Tech", 1.00, 1.00, 0.75, 0.75, 1.00),  # cs=0.9125 → T3 ✓
    ("v3_d5hi_018", "Produce a complete warehouse automation feasibility study including a current operations assessment, an automation technology comparison report with pricing from major robotics vendors, an ROI model, a facility layout redesign specification, an implementation timeline with resource requirements, and a risk mitigation plan covering operational continuity during the transition", "explicit", "Supply Chain", 1.00, 1.00, 1.00, 1.00, 1.00),  # cs=1.0 → T3 ✓
    ("v3_d5hi_019", "Develop a comprehensive zero-trust network architecture program including a current state security assessment, a network segmentation design document, identity and access management policy overhaul, microsegmentation implementation guides for both AWS and Azure environments, a monitoring and detection specification, and an executive compliance report mapping everything to NIST SP 800-207 with competitive benchmarking", "explicit", "Security", 1.00, 1.00, 1.00, 0.75, 1.00),  # cs=0.9625 → T3 ✓
    ("v3_d5hi_020", "Create a complete platform engineering program blueprint including a developer experience assessment report, an internal developer platform architecture specification, infrastructure-as-code template libraries, a service catalog design, a golden path documentation framework, and an adoption metrics dashboard specification, referencing platform engineering best practices from industry leaders", "explicit", "DevOps", 1.00, 1.00, 0.75, 0.50, 1.00),  # cs=0.875 → T3 ✓

    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH 3: BOUNDARY HARD T2 (15 prompts) — complexity 0.60–0.69
    # ═══════════════════════════════════════════════════════════════════════════

    ("v3_bt2_001", "I have uploaded our current Terraform configurations for our AWS networking setup. We need to redesign our VPC architecture to support three isolated environments with proper security group rules and NAT gateway configurations while keeping inter-service communication efficient", "explicit", "Cloud Infrastructure", 0.75, 0.75, 0.75, 0.25, 0.25),  # cs=0.2625+0.15+0.15+0.0375+0.025=0.625 → T2 ✓
    ("v3_bt2_002", "Our Kubernetes cluster needs a complete resource governance overhaul. Write a namespace-level resource quota policy and implement pod disruption budgets across all production workloads using the deployment configs I uploaded, and include a monitoring setup for tracking resource utilization compliance", "explicit", "DevOps", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.2625+0.15+0.15+0.0375+0.05=0.65 → T2 ✓
    ("v3_bt2_003", "Based on the attached data pipeline architecture, design a comprehensive data quality framework that integrates with our dbt models and Snowflake warehouse. It needs to cover freshness checks, schema validation, volume anomaly detection, and automated alerting for our data engineering and analytics teams", "explicit", "Data Engineering", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_004", "I have uploaded our employee handbook and current leave policies. We need to restructure our global PTO policy to comply with labor regulations across our US, UK and Germany offices while maintaining equity for remote workers and establishing a self-service portal for leave requests", "explicit", "HR Tech", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_005", "Our marketing team needs a unified customer data pipeline. Using the attached CRM schema and analytics platform configuration, design a real-time event streaming architecture that captures user interactions from our web app and mobile app and feeds them into our segmentation engine for personalized campaigns", "explicit", "Marketing Tech", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_006", "I have uploaded our current IAM policies and group membership exports. Perform a comprehensive access control audit and produce a least-privilege remediation plan that addresses the SOC 2 findings from our last assessment without disrupting developer productivity", "explicit", "Security", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_007", "Based on the uploaded cost allocation tags and department budget sheets, build a FinOps showback model that accurately attributes shared infrastructure costs to business units and provides monthly variance reports for the finance team", "explicit", "FinOps", 0.75, 0.75, 0.75, 0.25, 0.25),  # cs=0.625 → T2 ✓
    ("v3_bt2_008", "We need to integrate our Salesforce CPQ system with our legacy billing platform. Using the attached API documentation from both systems, design a bidirectional sync architecture that handles quote-to-cash workflow including error handling for price discrepancies and currency conversion edge cases", "explicit", "System Integration", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_009", "Our factory floor telemetry pipeline is dropping data during peak production hours. Using the attached sensor configuration and gateway logs, redesign the data collection architecture to handle burst traffic from 500 sensors with guaranteed delivery to our time-series database", "implicit", "IoT & Smart Factory", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_010", "Our AI model serving infrastructure needs to support multiple model versions simultaneously with canary deployments. We're running on Kubernetes and need to handle traffic splitting between model versions with automated rollback based on prediction accuracy drift thresholds", "implicit", "AI Governance", 0.75, 0.75, 0.75, 0.25, 0.25),  # cs=0.625 → T2 ✓
    ("v3_bt2_011", "our snowflake costs are completely out of control and the business intelligence team is running massive unoptimized queries every morning that take two hours to complete, we need to fix the warehouse sizing and query optimization before next month's budget review", "vague", "FinOps", 0.75, 0.75, 0.75, 0.00, 0.25),  # cs=0.5875 → T2 ✓ (lower boundary)
    ("v3_bt2_012", "I have uploaded our current Apache Kafka configuration and consumer group metrics. Our event processing pipeline needs to be redesigned to handle exactly-once semantics across three microservices while maintaining partition ordering guarantees during broker rebalances", "explicit", "Data Engineering", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_013", "Our supply chain planning team needs a demand sensing capability that incorporates point-of-sale data from our retail partners. Using the attached EDI configuration and inventory management schema, design a near-real-time demand signal processing pipeline with safety stock recalculation logic", "implicit", "Supply Chain", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_014", "Based on our attached incident response procedures and the recent penetration test findings, develop a comprehensive security incident playbook that covers detection, containment, eradication and recovery phases with specific runbooks for ransomware and data exfiltration scenarios", "explicit", "Security", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓
    ("v3_bt2_015", "We need to build an automated compliance evidence collection pipeline for our quarterly SOC 2 audits. Using the attached control matrix and system inventory, design an integration that pulls evidence from AWS CloudTrail, Okta logs, and GitHub audit logs into a centralized compliance repository", "explicit", "Security", 0.75, 0.75, 0.75, 0.25, 0.50),  # cs=0.65 → T2 ✓

    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH 4: BOUNDARY SOFT T3 (15 prompts) — complexity 0.70–0.80
    # ═══════════════════════════════════════════════════════════════════════════

    ("v3_bt3_001", "Based on the attached cloud spending reports, we need to compare our current AWS Reserved Instance portfolio against the latest Azure Reservation pricing and GCP Committed Use Discounts to find the optimal multi-cloud commitment strategy for the next fiscal year", "explicit", "FinOps", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.2625+0.15+0.15+0.1125+0.025=0.70 → T3 ✓ (barely)
    ("v3_bt3_002", "Our operations team needs to evaluate whether to migrate our on-premise Elasticsearch cluster to AWS OpenSearch or a managed Elastic Cloud deployment. We need a comparison of current vendor pricing, performance benchmarks, and a migration risk assessment covering data residency compliance for our EU customers", "implicit", "Cloud Infrastructure", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_003", "I have uploaded our current DevOps toolchain inventory and the engineering team's feedback survey results. Evaluate whether we should consolidate onto GitLab Ultimate or maintain our current multi-vendor setup of GitHub, Jenkins and ArgoCD, comparing the latest subscription pricing and factoring in developer productivity metrics", "implicit", "DevOps", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_004", "Using the attached warehouse operations data and current 3PL contracts, evaluate whether we should bring fulfillment in-house for our three highest-volume distribution centers or renegotiate with our current logistics providers, comparing current market rates for warehousing and last-mile delivery in each region", "implicit", "Supply Chain", 0.75, 0.75, 0.75, 0.75, 0.50),  # cs=0.2625+0.15+0.15+0.1125+0.05=0.725 → T3 ✓
    ("v3_bt3_005", "We discovered that our customer support team has been using an unapproved AI writing tool that processes customer conversation data through external APIs. We need to assess our regulatory exposure under GDPR and our existing SOC 2 commitments, and determine what technical controls to implement to prevent future unauthorized AI tool usage while comparing approved vendor alternatives", "implicit", "AI Governance", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_006", "Our manufacturing execution system data shows increasing quality defects correlated with a recent sensor firmware update. We need to cross-reference the sensor telemetry data with our quality management system records and the vendor's latest firmware release notes to determine whether to roll back or patch forward", "implicit", "IoT & Smart Factory", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_007", "Based on the attached candidate pipeline analytics and our current ATS configuration, evaluate whether our automated resume screening criteria are creating adverse impact against protected classes under EEOC guidelines, comparing our acceptance rates against industry benchmarks for similar roles", "implicit", "HR Tech", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_008", "the cto wants to know if we should build our own internal developer platform or buy one from backstage or cortex, need to figure out the total cost of ownership and what our competitors are doing about platform engineering", "vague", "DevOps", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_009", "our marketing attribution is completely broken since the ios privacy changes and the cmo is asking how competitors are handling it and what the latest industry solutions look like", "vague", "Marketing Tech", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_010", "multiple teams are deploying ai models without any governance and legal is worried about eu ai act compliance, we need to figure out what the regulatory landscape looks like and what controls to put in place before the next board meeting", "vague", "AI Governance", 1.00, 0.75, 0.75, 0.50, 0.25),  # cs=0.35+0.15+0.15+0.075+0.025=0.75 → T3 ✓
    ("v3_bt3_011", "Using the attached network topology diagrams and firewall rule exports, perform a security architecture review and identify all east-west traffic paths that violate our zero-trust segmentation policy, referencing NIST SP 800-207 guidelines and comparing our posture against current industry maturity benchmarks", "explicit", "Security", 0.75, 0.75, 0.75, 0.75, 0.50),  # cs=0.725 → T3 ✓
    ("v3_bt3_012", "Our data lakehouse on Databricks is becoming prohibitively expensive as data volumes grow. We need to evaluate whether to restructure our medallion architecture, migrate hot data to a cheaper storage tier, or consider splitting workloads between Databricks and a managed Snowflake instance, comparing current pricing for each approach", "implicit", "Data Engineering", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓
    ("v3_bt3_013", "The procurement team wants to consolidate our SaaS vendor portfolio. Based on the attached vendor inventory and contract terms, identify overlapping tools across departments and provide a rationalization recommendation that factors in current renewal pricing and switching costs, benchmarked against what comparable companies spend", "implicit", "General Enterprise", 0.75, 0.75, 0.75, 0.75, 0.50),  # cs=0.725 → T3 ✓
    ("v3_bt3_014", "Based on the uploaded Salesforce and HubSpot data schemas, we need to unify our customer data across both CRMs into a single golden record while handling duplicate resolution and ensuring GDPR-compliant data handling, comparing CDP vendor solutions at current market rates", "implicit", "System Integration", 0.75, 0.75, 0.75, 0.75, 0.50),  # cs=0.725 → T3 ✓
    ("v3_bt3_015", "Our last-mile delivery costs are spiraling and customer complaints about late deliveries have doubled. We need to analyze our routing efficiency against current fuel costs and competitor delivery SLAs, and determine whether to invest in route optimization software or switch to a different 3PL provider", "implicit", "Supply Chain", 0.75, 0.75, 0.75, 0.75, 0.25),  # cs=0.70 → T3 ✓

    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH 5: T3 WITH EXPLICIT D4 TEXT SIGNALS (15 prompts)
    # Market research, vendor pricing, competitive analysis, industry benchmarks
    # ═══════════════════════════════════════════════════════════════════════════

    ("v3_d4hi_001", "Conduct a comprehensive market research analysis comparing the latest enterprise LLM API pricing from OpenAI, Anthropic, Google and Cohere, and produce a vendor selection matrix that factors in data privacy certifications, throughput benchmarks, and total cost of ownership projections for our customer service automation use case", "explicit", "AI Governance", 1.00, 1.00, 1.00, 1.00, 0.75),  # cs=0.9375 → T3 ✓
    ("v3_d4hi_002", "We need a competitive analysis of our cloud infrastructure costs against industry benchmarks published by Gartner and Flexera. Pull the latest cloud cost management survey data and compare our per-workload spending on compute, storage and networking against the median for companies in our revenue bracket", "explicit", "FinOps", 0.75, 0.75, 0.75, 1.00, 0.50),  # cs=0.2625+0.15+0.15+0.15+0.05=0.7625 → T3 ✓
    ("v3_d4hi_003", "Research the current pricing models and compliance certifications for the top five identity governance vendors including SailPoint, Saviynt, CyberArk and Okta Identity Governance, and produce a procurement recommendation that aligns with our NIST 800-53 access control requirements and our three-year security roadmap", "explicit", "Security", 1.00, 1.00, 1.00, 1.00, 0.75),  # cs=0.9375 → T3 ✓
    ("v3_d4hi_004", "Our supply chain team needs a market research report on alternative raw material suppliers in Southeast Asia. Compare the current pricing, lead times, minimum order quantities and sustainability certifications of at least five potential suppliers against our existing contracts with our Chinese manufacturers", "explicit", "Supply Chain", 0.75, 1.00, 0.75, 1.00, 0.75),  # cs=0.2625+0.20+0.15+0.15+0.075=0.8375 → T3 ✓
    ("v3_d4hi_005", "Produce an industry trends analysis on the adoption of generative AI in enterprise customer support. Reference the latest analyst reports from Gartner, Forrester and McKinsey, compare vendor pricing for conversational AI platforms, and assess how our planned deployment compares against competitor implementations in our vertical", "explicit", "Competitive Intelligence", 1.00, 1.00, 1.00, 1.00, 0.75),  # cs=0.9375 → T3 ✓
    ("v3_d4hi_006", "We need to evaluate the latest real-time pricing for GPU cloud instances across AWS, Azure, GCP and CoreWeave for our ML training workloads. Compare spot vs on-demand vs reserved pricing tiers, analyze regional price variations, and recommend an optimal multi-cloud GPU procurement strategy that minimizes cost while meeting our data residency requirements", "explicit", "FinOps", 1.00, 1.00, 1.00, 1.00, 0.50),  # cs=0.35+0.20+0.20+0.15+0.05=0.95 → T3 ✓
    ("v3_d4hi_007", "Research current market rates for cybersecurity insurance and produce a cost-benefit analysis for our organization. Compare policy terms from at least four major cyber insurance providers, factor in our current risk posture assessment, and determine the optimal coverage level given recent industry claims data and premium trends", "implicit", "Security", 0.75, 0.75, 0.75, 1.00, 0.50),  # cs=0.7625 → T3 ✓
    ("v3_d4hi_008", "The HR director wants a competitive compensation benchmarking report for our engineering roles across three markets. Research current salary survey data from Radford, Levels.fyi and Glassdoor, compare our total compensation packages against FAANG and Series B-D startups, and produce recommendations for adjusting our pay bands to reduce attrition", "implicit", "HR Tech", 0.75, 1.00, 0.75, 1.00, 0.75),  # cs=0.8375 → T3 ✓
    ("v3_d4hi_009", "our cfo wants to know how our cloud spend per employee compares against our direct competitors and industry averages, and whether we should be looking at different pricing models from the cloud vendors", "vague", "FinOps", 0.75, 0.75, 0.75, 1.00, 0.25),  # cs=0.2625+0.15+0.15+0.15+0.025=0.7375 → T3 ✓
    ("v3_d4hi_010", "marketing wants to know what our competitors are spending on paid social and whether the latest industry benchmark reports show different channel allocation strategies we should consider", "vague", "Marketing Tech", 0.75, 0.75, 0.75, 1.00, 0.25),  # cs=0.7375 → T3 ✓
    ("v3_d4hi_011", "Research the latest regulatory guidance on cross-border data transfers between the EU and US following the Data Privacy Framework adequacy decision. Analyze how this affects our current data processing architecture across AWS regions, compare compliance approaches used by competitors in our industry, and produce updated standard contractual clauses for our vendor agreements", "explicit", "Regulatory Compliance", 1.00, 1.00, 1.00, 1.00, 0.75),  # cs=0.9375 → T3 ✓
    ("v3_d4hi_012", "We need a comprehensive market analysis of warehouse automation technologies. Research the current pricing and capabilities of autonomous mobile robots from vendors like Locus Robotics, 6 River Systems and Fetch Robotics, compare against traditional conveyor systems, and model the five-year total cost of ownership for our three largest distribution centers", "explicit", "Supply Chain", 1.00, 1.00, 1.00, 1.00, 0.75),  # cs=0.9375 → T3 ✓
    ("v3_d4hi_013", "The CTO wants a technology radar assessment for our engineering organization. Survey the latest industry trends in platform engineering, AI-assisted development and cloud-native architecture, benchmark our technology stack maturity against industry peers, and produce a strategic investment recommendation for the next two fiscal years", "implicit", "DevOps", 0.75, 0.75, 0.75, 1.00, 0.50),  # cs=0.7625 → T3 ✓
    ("v3_d4hi_014", "Analyze current market pricing for managed Kubernetes services across AWS EKS, Azure AKS and Google GKE including support tiers, add-on pricing and egress costs. Compare against self-managed alternatives and produce a five-year TCO projection for our microservices platform running 200 pods across three regions", "explicit", "Cloud Infrastructure", 1.00, 1.00, 1.00, 1.00, 0.50),  # cs=0.95 → T3 ✓
    ("v3_d4hi_015", "Research the latest industry best practices and vendor offerings for real-time fraud detection in payment processing. Compare pricing and detection accuracy benchmarks for Stripe Radar, Sift and Forter, evaluate their integration requirements with our current payment gateway, and recommend a solution that balances false positive rates against chargeback prevention", "implicit", "System Integration", 0.75, 1.00, 0.75, 1.00, 0.50),  # cs=0.8375 → T3 ✓

    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH 6: T1 BALANCE (20 prompts) — simple, single-topic questions
    # ═══════════════════════════════════════════════════════════════════════════

    ("v3_t1_001", "what is the difference between a docker image and a docker container", "explicit", "DevOps", 0.25, 0.50, 0.25, 0.00, 0.00),  # cs=0.2375 → T1 ✓
    ("v3_t1_002", "how do I create a read replica in aws rds", "explicit", "Cloud Infrastructure", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_003", "what is the maximum file size for a single s3 put request", "explicit", "Cloud Infrastructure", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_004", "explain what a kubernetes pod is in simple terms", "vague", "DevOps", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_005", "how do I check the current utilization of an ec2 instance from the cli", "explicit", "Cloud Infrastructure", 0.00, 0.25, 0.25, 0.00, 0.00),  # cs=0.10 → T1 ✓
    ("v3_t1_006", "what does the 429 http status code mean", "vague", "DevOps", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_007", "how do I reset a forgotten admin password in our okta tenant", "explicit", "Security", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_008", "what is a materialized view in snowflake and when should I use one", "explicit", "Data Engineering", 0.25, 0.50, 0.25, 0.00, 0.00),  # cs=0.2375 → T1 ✓
    ("v3_t1_009", "our jenkins server ran out of disk space again", "vague", "DevOps", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_010", "what is the difference between symmetric and asymmetric encryption", "explicit", "Security", 0.25, 0.50, 0.25, 0.00, 0.00),  # cs=0.2375 → T1 ✓
    ("v3_t1_011", "how many vacation days do I have left this year", "vague", "General Enterprise", 0.00, 0.00, 0.00, 0.00, 0.00),  # cs=0.00 → T1 ✓
    ("v3_t1_012", "what is the difference between etl and elt", "explicit", "Data Engineering", 0.25, 0.50, 0.25, 0.00, 0.00),  # cs=0.2375 → T1 ✓
    ("v3_t1_013", "the attached terraform plan output is showing a destroy action on our production database, is this intentional", "implicit", "Cloud Infrastructure", 0.00, 0.50, 0.00, 0.25, 0.25),  # cs=0.175 → T1 ✓
    ("v3_t1_014", "how do I add a new column to an existing table in bigquery without dropping the table", "explicit", "Data Engineering", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_015", "whats the standard process for submitting a purchase order in our sap system", "vague", "Supply Chain", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_016", "looking at the error log I pasted, why is our airflow dag failing with a connection timeout to the postgres source", "implicit", "Data Engineering", 0.00, 0.50, 0.00, 0.25, 0.25),  # cs=0.175 → T1 ✓
    ("v3_t1_017", "what does cpm stand for in digital advertising", "vague", "Marketing Tech", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_018", "how do I enable mfa for my aws root account", "explicit", "Security", 0.00, 0.25, 0.00, 0.00, 0.00),  # cs=0.05 → T1 ✓
    ("v3_t1_019", "what is the difference between a load balancer and a reverse proxy", "explicit", "Cloud Infrastructure", 0.25, 0.50, 0.25, 0.00, 0.00),  # cs=0.2375 → T1 ✓
    ("v3_t1_020", "the grafana dashboard is not showing any data for the last two hours, based on the screenshot I attached what could be wrong", "implicit", "DevOps", 0.00, 0.50, 0.00, 0.25, 0.25),  # cs=0.175 → T1 ✓

    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH 7: T2 EXPLICIT/IMPLICIT VARIETY (15 prompts) — fill gaps
    # ═══════════════════════════════════════════════════════════════════════════

    ("v3_t2v_001", "Based on the attached network diagram and current firewall rules, design a DMZ architecture that isolates our public-facing web servers from our internal application tier while maintaining the API connectivity required for our mobile application backend", "explicit", "Cloud Infrastructure", 0.50, 0.50, 0.50, 0.25, 0.25),  # cs=0.175+0.10+0.10+0.0375+0.025=0.4375 → T2 ✓
    ("v3_t2v_002", "Our Airflow DAGs are failing because upstream source tables in Salesforce change their schemas without notice. We need to build a schema drift detection layer that alerts the data team and automatically adjusts our staging models before the downstream transforms break", "implicit", "Data Engineering", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_t2v_003", "Create a runbook for our on-call team that covers the top ten most common production incidents based on the attached PagerDuty incident history, including diagnosis steps, resolution procedures and escalation paths for each scenario", "explicit", "DevOps", 0.50, 0.50, 0.50, 0.25, 0.50),  # cs=0.175+0.10+0.10+0.0375+0.05=0.4625 → T2 ✓
    ("v3_t2v_004", "Using the attached employee engagement survey results, identify the key drivers of attrition in our engineering department and propose specific interventions for each problem area including concrete timeline and success metrics", "implicit", "HR Tech", 0.75, 0.75, 0.50, 0.00, 0.25),  # cs=0.5375 → T2 ✓
    ("v3_t2v_005", "Our Snowflake virtual warehouse auto-suspend is set too aggressively and queries are hitting cold start penalties every time. We need to optimize our warehouse sizing and scheduling to balance cost against the query performance SLAs the analytics team requires", "implicit", "FinOps", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_t2v_006", "Design a CI/CD pipeline for our monorepo that supports incremental builds across six microservices, with proper caching strategies and automated integration testing between dependent services using the GitHub Actions workflow structure I uploaded", "explicit", "DevOps", 0.50, 0.50, 0.75, 0.25, 0.25),  # cs=0.175+0.10+0.15+0.0375+0.025=0.4875 → T2 ✓
    ("v3_t2v_007", "Write a comprehensive API rate limiting and throttling strategy for our public REST API that handles 50000 requests per minute. Include token bucket configuration, customer-tier-based quotas, and graceful degradation behavior during traffic spikes", "explicit", "System Integration", 0.50, 0.50, 0.50, 0.00, 0.25),  # cs=0.40 → T2 ✓
    ("v3_t2v_008", "Our SCADA system needs to be segmented from our corporate IT network after the latest OT security assessment. Using the attached network topology and device inventory, design a Purdue model implementation that maintains necessary data flows from the factory floor to our cloud analytics platform", "implicit", "IoT & Smart Factory", 0.75, 0.75, 0.75, 0.25, 0.25),  # cs=0.625 → T2 ✓
    ("v3_t2v_009", "Based on the uploaded Google Analytics 4 configuration, redesign our event tracking taxonomy to properly capture the full e-commerce funnel from product view through checkout completion including custom dimensions for A/B test variants and user cohort segments", "explicit", "Marketing Tech", 0.50, 0.50, 0.50, 0.25, 0.25),  # cs=0.4375 → T2 ✓
    ("v3_t2v_010", "Our Terraform state files are getting corrupted during concurrent applies from multiple team members. We need to implement a proper state locking strategy with remote backends and establish a GitOps workflow that prevents conflicts while supporting our multi-environment deployment model", "implicit", "DevOps", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_t2v_011", "I have uploaded our current AWS Lambda function configurations and CloudWatch metrics. Several functions are hitting timeout limits and memory constraints during month-end processing. We need to optimize the function architecture and determine which workloads should be moved to ECS Fargate for better resource control", "implicit", "Cloud Infrastructure", 0.50, 0.50, 0.50, 0.25, 0.25),  # cs=0.4375 → T2 ✓
    ("v3_t2v_012", "Design a data retention and archival policy for our data warehouse that balances regulatory requirements for seven-year record keeping against storage costs. Include automated lifecycle rules for moving cold data to cheaper tiers and a metadata catalog for archived datasets", "explicit", "Data Engineering", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
    ("v3_t2v_013", "Our inventory management system is not properly accounting for returns and the warehouse counts are drifting from the ERP records. Using the attached reconciliation reports, identify the root causes of the inventory discrepancy and design a corrective workflow that prevents future mismatches", "implicit", "Supply Chain", 0.75, 0.50, 0.50, 0.25, 0.25),  # cs=0.525 → T2 ✓
    ("v3_t2v_014", "Create a formal data classification policy for our organization that defines sensitivity levels, handling requirements and access controls for each classification tier, aligned with our upcoming SOC 2 Type II audit requirements and the data types documented in the attached data catalog", "explicit", "Security", 0.75, 0.75, 0.50, 0.25, 0.25),  # cs=0.575 → T2 ✓
    ("v3_t2v_015", "Our machine learning model retraining pipeline runs weekly but often produces worse models than the current production version. We need to implement an automated model evaluation gate that compares new model performance against the production baseline before promoting, including drift detection and A/B traffic splitting capabilities", "implicit", "AI Governance", 0.75, 0.50, 0.50, 0.00, 0.25),  # cs=0.4875 → T2 ✓
]

# ── Build DataFrame from new rows ────────────────────────────────────────────
columns = ['id', 'prompt', 'phrasing_style', 'domain', 'd1', 'd2', 'd3', 'd4', 'd5']
df_new = pd.DataFrame(new_rows, columns=columns)

# ── Validate all scores ──────────────────────────────────────────────────────
def compute_cs(row):
    return row['d1']*0.35 + row['d2']*0.20 + row['d3']*0.20 + row['d4']*0.15 + row['d5']*0.10

def get_tier(cs):
    if cs < 0.40: return 'T1'
    elif cs < 0.70: return 'T2'
    else: return 'T3'

df_new['cs'] = df_new.apply(compute_cs, axis=1)
df_new['tier'] = df_new['cs'].apply(get_tier)

# Verify tier assignments
print(f"\nNew prompts added: {len(df_new)}")
print(f"\nNew prompt tier distribution:")
print(df_new['tier'].value_counts().sort_index())
print(f"\nNew prompt phrasing_style distribution:")
print(df_new.groupby(['tier', 'phrasing_style']).size().unstack(fill_value=0))
print(f"\nD5 values in new data:")
print(df_new['d5'].value_counts().sort_index())
print(f"\nD4 values in new T3 data:")
print(df_new[df_new['tier']=='T3']['d4'].value_counts().sort_index())

# Check for ID uniqueness
df_new.drop(columns=['cs', 'tier'], inplace=True)

# ── Merge with original ─────────────────────────────────────────────────────
df_merged = pd.concat([df_orig, df_new], ignore_index=True)
assert df_merged['id'].is_unique, "Duplicate IDs found!"

# ── Save ─────────────────────────────────────────────────────────────────────
df_merged.to_csv('dataset_prompt_profiling_v2.csv', index=False)
print(f"\nMerged dataset: {len(df_merged)} rows → dataset_prompt_profiling_v2.csv")

# Final distribution check
df_merged['cs'] = df_merged.apply(compute_cs, axis=1)
df_merged['tier'] = df_merged['cs'].apply(get_tier)
print(f"\nFinal tier distribution:")
print(df_merged['tier'].value_counts().sort_index())
print(f"\nFinal phrasing by tier:")
print(pd.crosstab(df_merged['tier'], df_merged['phrasing_style']))
print(f"\nFinal D4 by tier:")
print(df_merged.groupby('tier')['d4'].value_counts().unstack(fill_value=0))
print(f"\nFinal D5 value counts:")
print(df_merged['d5'].value_counts().sort_index())
