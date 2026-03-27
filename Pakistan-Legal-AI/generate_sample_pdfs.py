"""
Generate sample Pakistani law PDF files for testing the ingestion pipeline.
Run once: python generate_sample_pdfs.py
"""

from fpdf import FPDF
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def create_pdf(filename, title, sections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 15, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    # Sections
    pdf.set_font("Helvetica", "", 11)
    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(4)
    
    filepath = os.path.join(DATA_DIR, filename)
    pdf.output(filepath)
    print(f"  Created: {filepath}")

# ─── PDF 1: Constitution of Pakistan (Fundamental Rights) ───────────────────

create_pdf("constitution_fundamental_rights.pdf", 
    "Constitution of Pakistan - Fundamental Rights",
    [
        ("Article 8 - Laws inconsistent with fundamental rights",
         "Any law, or any custom or usage having the force of law, in so far as it is inconsistent with the rights conferred by this Chapter, shall, to the extent of such inconsistency, be void."),
        
        ("Article 9 - Security of Person",
         "No person shall be deprived of life or liberty save in accordance with law. Every citizen shall have the right to be protected by the State from any violence or threat of violence."),
        
        ("Article 10 - Safeguards as to Arrest and Detention",
         "No person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest, nor shall he be denied the right to consult and be defended by a legal practitioner of his choice. Every person who is arrested and detained in custody shall be produced before a magistrate within a period of twenty-four hours of such arrest."),
        
        ("Article 10A - Right to Fair Trial",
         "For the determination of his civil rights and obligations or in any criminal charge against him a person shall be entitled to a fair trial and due process."),
        
        ("Article 11 - Slavery and Forced Labour",
         "Slavery is non-existent and forbidden and no law shall permit or facilitate its introduction into Pakistan in any form. All forms of forced labour and traffic in human beings are prohibited. No child below the age of fourteen years shall be engaged in any factory or mine or any other hazardous employment."),
        
        ("Article 14 - Inviolability of Dignity of Man",
         "The dignity of man and, subject to law, the privacy of home shall be inviolable. No person shall be subjected to torture for the purpose of extracting evidence."),
        
        ("Article 18 - Freedom of Trade, Business or Profession",
         "Subject to such qualifications, if any, as may be prescribed by law, every citizen shall have the right to enter upon any lawful profession or occupation, and to conduct any lawful trade or business."),
        
        ("Article 19 - Freedom of Speech",
         "Every citizen shall have the right to freedom of speech and expression, and there shall be freedom of the press, subject to any reasonable restrictions imposed by law in the interest of the glory of Islam or the integrity, security or defence of Pakistan."),
        
        ("Article 19A - Right to Information",
         "Every citizen shall have the right to have access to information in all matters of public importance subject to regulation and reasonable restrictions imposed by law."),
        
        ("Article 20 - Freedom of Religion",
         "Subject to law, public order and morality, every citizen shall have the right to profess, practice and propagate his religion; and every religious denomination shall have the right to establish, maintain and manage its religious institutions."),
        
        ("Article 22 - Safeguards as to Educational Institutions",
         "No person attending any educational institution shall be required to receive religious instruction, or take part in any religious ceremony, or attend religious worship, if such instruction, ceremony or worship relates to a religion other than his own."),
        
        ("Article 25 - Equality of Citizens",
         "All citizens are equal before law and are entitled to equal protection of law. There shall be no discrimination on the basis of sex alone. Nothing in this Article shall prevent the State from making any special provision for the protection of women and children."),
        
        ("Article 25A - Right to Education",
         "The State shall provide free and compulsory education to all children of the age of five to sixteen years in such manner as may be determined by law."),
    ]
)

# ─── PDF 2: Pakistan Labour Laws ────────────────────────────────────────────

create_pdf("pakistan_labour_laws.pdf",
    "Pakistan Labour Laws - Consolidated Summary",
    [
        ("Minimum Wage - Federal",
         "The minimum wage in Pakistan is set by provincial governments. As of 2024, the federal minimum wage is PKR 32,000 per month for unskilled workers. Skilled workers may receive higher wages as determined by provincial wage boards. Employers who pay below the minimum wage face penalties under the Minimum Wages Ordinance 1961."),
        
        ("Working Hours and Overtime",
         "No adult worker shall be required or allowed to work in any establishment in excess of nine hours a day and forty-eight hours a week. Any work done beyond these limits constitutes overtime, which must be compensated at a rate not less than double the ordinary rate of wages. Spread over of working hours shall not exceed ten and a half hours in any day."),
        
        ("Annual Leave and Holidays",
         "Every worker who has completed a period of twelve months continuous service in an establishment shall be allowed, during the subsequent period of twelve months, leave with full pay for a period of not less than fourteen consecutive days. In addition, workers are entitled to casual leave of ten days and sick leave of sixteen days with half pay per year."),
        
        ("Employment of Women",
         "No woman shall be employed in any establishment during the night except in such establishments as the Government may prescribe. Women are entitled to maternity leave of twelve weeks (six weeks before and six weeks after delivery) with full pay. Dismissal of a woman during maternity leave is prohibited."),
        
        ("Employment of Children",
         "No child who has not completed his fourteenth year of age shall be required or allowed to work in any establishment. Adolescents between 14 and 18 years may work with restrictions: no night work, no overtime, no hazardous occupations, and maximum 7 hours of work per day."),
        
        ("Industrial Relations",
         "Workers have the right to form and join trade unions. Employers cannot terminate, dismiss, or otherwise punish a worker for being a member of a trade union. Collective bargaining agents are determined through secret ballot elections among workers. Industrial disputes that cannot be resolved through negotiation are referred to labour courts."),
        
        ("Termination and Severance",
         "An employer may terminate a permanent worker by giving one month notice or wages in lieu thereof. Workers who have been in continuous service for more than one year are entitled to gratuity at the rate of thirty days wages for every completed year of service. Wrongful termination may be challenged through the labour court within thirty days."),
        
        ("Occupational Safety and Health",
         "Every establishment shall ensure the safety, health and welfare of all workers. This includes adequate ventilation, lighting, sanitary facilities, first aid equipment, and fire safety measures. Factories with more than 250 workers must employ a welfare officer. Industrial accidents must be reported to the relevant inspector within 24 hours."),
    ]
)

# ─── PDF 3: Pakistan Tax Laws ───────────────────────────────────────────────

create_pdf("pakistan_tax_laws.pdf",
    "Pakistan Tax Laws - Income Tax and Sales Tax",
    [
        ("Income Tax - Salaried Individuals (Tax Year 2024)",
         "Pakistan uses a MARGINAL tax system for salaried individuals. The tax brackets are as follows: Income up to PKR 600,000 per annum is exempt from tax (0%). For income exceeding PKR 600,000 but not exceeding PKR 1,200,000, tax is 5% of the amount exceeding PKR 600,000. For income exceeding PKR 1,200,000 but not exceeding PKR 2,400,000, tax is PKR 30,000 plus 10% of the amount exceeding PKR 1,200,000. For income exceeding PKR 2,400,000 but not exceeding PKR 3,600,000, tax is PKR 150,000 plus 15% of the amount exceeding PKR 2,400,000."),
        
        ("Income Tax - Business Individuals",
         "For business individuals (non-salaried), the tax rates are generally higher. Income up to PKR 600,000 per annum is exempt. For income exceeding PKR 600,000 but not exceeding PKR 800,000, tax is 15% of the amount exceeding PKR 600,000. Tax filers who are on the Active Taxpayer List (ATL) enjoy reduced withholding tax rates compared to non-filers."),
        
        ("Sales Tax",
         "The standard rate of sales tax in Pakistan is 18% on the value of taxable supplies made in Pakistan (increased from 17% in Budget 2024-25). Certain essential items are zero-rated or exempt, including basic food items (wheat, rice, pulses), medicines, and educational materials. Retailers with annual turnover exceeding PKR 10 million must register for sales tax."),
        
        ("Withholding Tax",
         "Pakistan employs an extensive withholding tax regime. Employers must deduct income tax from salaries. Banks deduct withholding tax on profit/interest. The rates differ for filers and non-filers. Non-filers face significantly higher withholding tax rates as an incentive to file tax returns."),
        
        ("Property Tax and Capital Gains",
         "Capital gains on sale of immovable property held for less than one year are taxed at 15%. Property held between one and two years is taxed at 12.5%, between two and three years at 10%, between three and four years at 7.5%, between four and five years at 5%, and between five and six years at 2.5%. Property held for more than six years is exempt from capital gains tax."),
        
        ("Tax Filing Requirements",
         "Every individual whose income exceeds PKR 600,000 per annum is required to file an income tax return. The deadline for filing returns is September 30 of each year for salaried individuals. Late filing attracts a penalty of PKR 1,000 to PKR 50,000. Non-filing can result in being placed on the Non-Active Taxpayer List, which attracts higher withholding tax rates on all financial transactions."),
    ]
)

# ─── PDF 4: Tenant and Property Rights ──────────────────────────────────────

create_pdf("tenant_property_rights.pdf",
    "Pakistan Tenant and Property Rights",
    [
        ("Tenant Protection",
         "A landlord cannot evict a tenant without proper legal notice. Under the Punjab Rented Premises Act 2009 and similar provincial laws, a minimum of one month written notice is required before eviction proceedings can begin. The notice must specify the grounds for eviction. A tenant can only be evicted on grounds specified in the law, such as non-payment of rent, subletting without permission, or the landlord's personal need."),
        
        ("Rent Increase Limitations",
         "Landlords cannot increase rent arbitrarily. Under provincial rent restriction laws, rent can only be increased once per year, and the increase is generally limited to 10% of the existing rent. Any rent increase must be communicated through written notice at least three months before it takes effect."),
        
        ("Security Deposit",
         "The security deposit collected by a landlord shall not exceed the equivalent of two months rent. The deposit must be returned within thirty days of the tenant vacating the premises, minus any legitimate deductions for damages beyond normal wear and tear. The landlord must provide an itemized list of deductions."),
        
        ("Tenant Obligations",
         "Tenants are obligated to pay rent on time as agreed in the tenancy agreement. The tenant must maintain the premises in reasonable condition and not cause damage beyond normal wear and tear. Subletting or assigning the tenancy without the landlord's written consent is prohibited. The tenant must allow the landlord reasonable access for inspection with prior notice of at least 24 hours."),
        
        ("Dispute Resolution",
         "Disputes between landlords and tenants are adjudicated by Rent Tribunals or Civil Courts depending on the jurisdiction. In Punjab, the Rent Tribunal has exclusive jurisdiction over eviction and rent disputes. Cases must be decided within 90 days. Appeals can be filed within 30 days of the Tribunal's decision."),
    ]
)

print(f"\n✅ Generated 4 sample PDF files in {DATA_DIR}/")
print("Run 'python ingest.py' to build the vector database.")
