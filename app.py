"""
PI Hunter - NIH Grant Recruitment Value Estimator

Searches NIH Reporter for a researcher's active grants and estimates
the recruitment value (portable funding) they could bring.
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go

# NIH Reporter API configuration
NIH_API_URL = "https://api.reporter.nih.gov/v2/projects/search"

# Grant type portability scores (0-1 scale)
# Based on how likely/easy it is for a grant to transfer with a PI
PORTABILITY_SCORES = {
    # Research grants - highly portable
    "R01": 0.95,
    "R21": 0.95,
    "R03": 0.95,
    "R15": 0.95,  # AREA grants
    "R35": 0.90,  # MIRA grants
    "R33": 0.90,
    "R34": 0.90,
    "R37": 0.95,  # MERIT awards
    "R56": 0.90,

    # Career development - follows the person
    "K01": 0.85,
    "K08": 0.85,
    "K22": 0.85,
    "K23": 0.85,
    "K25": 0.85,
    "K99": 0.95,  # Pathway to independence
    "R00": 0.95,  # Second phase of K99

    # Director's awards - portable
    "DP1": 0.90,
    "DP2": 0.90,
    "DP5": 0.90,

    # Fellowships - follow the person
    "F30": 0.95,
    "F31": 0.95,
    "F32": 0.95,
    "F33": 0.95,

    # Cooperative agreements - varies significantly
    "U01": 0.50,  # Research project cooperative
    "U19": 0.30,  # Research program cooperative
    "U24": 0.25,  # Resource-related
    "U54": 0.20,  # Specialized center cooperative
    "UG3": 0.60,
    "UH3": 0.60,

    # Center grants - mostly institutional
    "P01": 0.30,  # Research program project
    "P20": 0.15,
    "P30": 0.10,  # Center core grants
    "P50": 0.15,  # Specialized centers
    "P51": 0.05,  # Primate centers

    # Training grants - institutional
    "T32": 0.05,
    "T34": 0.05,
    "T35": 0.05,

    # SBIR/STTR - company based
    "R41": 0.20,
    "R42": 0.20,
    "R43": 0.20,
    "R44": 0.20,
}

DEFAULT_PORTABILITY = 0.50


def get_portability_score(activity_code: str) -> float:
    """Get portability score for a grant type."""
    if not activity_code:
        return DEFAULT_PORTABILITY
    # Extract base activity code (e.g., "R01" from "R01A")
    base_code = ''.join(c for c in activity_code[:3] if c.isalnum())
    return PORTABILITY_SCORES.get(base_code, DEFAULT_PORTABILITY)


def get_portability_category(score: float) -> str:
    """Categorize portability score."""
    if score >= 0.85:
        return "Highly Portable"
    elif score >= 0.60:
        return "Likely Portable"
    elif score >= 0.40:
        return "Partially Portable"
    else:
        return "Unlikely to Transfer"


def calculate_time_remaining_factor(project_end: str, project_start: str) -> float:
    """
    Calculate what fraction of the project period remains.
    Returns a value between 0 and 1.
    """
    if not project_end:
        return 0.5  # Default if no end date

    try:
        end_date = datetime.strptime(project_end, "%Y-%m-%d").date()
        start_date = datetime.strptime(project_start, "%Y-%m-%d").date() if project_start else None
        today = date.today()

        if end_date <= today:
            return 0.0  # Project has ended

        # Calculate remaining time
        days_remaining = (end_date - today).days

        if start_date:
            total_days = (end_date - start_date).days
            if total_days > 0:
                return min(1.0, days_remaining / total_days)

        # If no start date, estimate based on typical 5-year grant
        typical_grant_days = 5 * 365
        return min(1.0, days_remaining / typical_grant_days)

    except (ValueError, TypeError):
        return 0.5


def normalize_name(name: str) -> str:
    """Normalize a name for comparison."""
    return ' '.join(name.lower().split())


def names_match(search_name: str, pi_full_name: str) -> bool:
    """
    Check if search name matches a PI's full name.
    Handles various formats: "First Last", "Last, First", etc.
    Requires exact last name match and close first name match.
    """
    search_parts = normalize_name(search_name).split()
    pi_normalized = normalize_name(pi_full_name)
    pi_parts = pi_normalized.replace(',', ' ').split()

    if len(search_parts) < 2:
        # Single name - check if it's exactly in the PI name
        return search_parts[0] in pi_parts

    search_first = search_parts[0]
    search_last = search_parts[-1]

    # Last name must match exactly
    last_name_match = search_last in pi_parts

    if not last_name_match:
        return False

    # First name matching - stricter rules:
    # 1. Exact match, OR
    # 2. PI has initial/short name that matches start of search (e.g., "H" matches "Hua"), OR
    # 3. Search has initial that matches start of PI name (e.g., "H" matches "Hua")
    first_name_match = False
    for pi_part in pi_parts:
        if pi_part == search_last:
            continue  # Skip the last name

        # Exact match
        if pi_part == search_first:
            first_name_match = True
            break

        # Short PI name is prefix of search name (PI has initial like "H" or "H.")
        # Only allow if PI part is 1-2 chars (likely an initial)
        if len(pi_part) <= 2 and search_first.startswith(pi_part.rstrip('.')):
            first_name_match = True
            break

        # Short search name is prefix of PI name (user typed initial)
        # Only allow if search is 1-2 chars
        if len(search_first) <= 2 and pi_part.startswith(search_first):
            first_name_match = True
            break

    return first_name_match


def get_base_project_number(project_num: str) -> str:
    """
    Extract the base project number without the year suffix and application type prefix.

    NIH project numbers format: [app_type][activity_code][IC][serial]-[year][suffix]
    E.g., '5R01AG078154-04' -> 'R01AG078154'
          '1U24MH136069-01' -> 'U24MH136069'
          '3R01AG078154-04S1' -> 'R01AG078154' (supplement)

    Application type prefixes (1, 2, 3, 4, 5, 6, 7, 8, 9) indicate:
    1=New, 2=Renewal, 3=Supplement, 4=Extension, 5=Non-competing continuation, etc.
    """
    if not project_num:
        return project_num

    # Remove the -XX or -XXS1 suffix (year/supplement indicator)
    if '-' in project_num:
        base = project_num.rsplit('-', 1)[0]
    else:
        base = project_num

    # Remove leading application type digit (1-9)
    if base and base[0].isdigit():
        base = base[1:]

    return base


def deduplicate_grants(grants: list) -> list:
    """
    Remove duplicate grant years, keeping only the most recent year for each grant.
    Multi-year grants appear as separate records (e.g., -01, -02, -03).
    We only want to count the current/latest year.
    """
    # Group by base project number
    grant_dict = {}

    for grant in grants:
        project_num = grant.get("project_num", "")
        base_num = get_base_project_number(project_num)
        fiscal_year = grant.get("fiscal_year", 0)

        # Keep the grant with the highest fiscal year for each base project
        if base_num not in grant_dict:
            grant_dict[base_num] = grant
        else:
            existing_fy = grant_dict[base_num].get("fiscal_year", 0)
            if fiscal_year > existing_fy:
                grant_dict[base_num] = grant

    return list(grant_dict.values())


def search_nih_reporter(pi_name: str, include_active_only: bool = True) -> list:
    """
    Search NIH Reporter for grants by PI name.
    Uses the API's name search then filters for exact matches.
    Deduplicates multi-year grants to only include the latest year.
    """
    # Parse the name for better API query
    name_parts = pi_name.strip().split()

    # Build the search criteria - use first and last name if available
    if len(name_parts) >= 2:
        pi_search = {
            "first_name": name_parts[0],
            "last_name": name_parts[-1]
        }
    else:
        pi_search = {"any_name": pi_name}

    criteria = {
        "pi_names": [pi_search],
        "exclude_subprojects": True,
    }

    if include_active_only:
        criteria["is_active"] = True

    payload = {
        "criteria": criteria,
        "offset": 0,
        "limit": 500,
        "sort_field": "project_start_date",
        "sort_order": "desc"
    }

    try:
        response = requests.post(
            NIH_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        # Post-filter to ensure the target PI is actually on the grant
        filtered_results = []
        for grant in results:
            pis = grant.get("principal_investigators", [])
            for pi in pis:
                pi_full_name = pi.get("full_name", "")
                if names_match(pi_name, pi_full_name):
                    filtered_results.append(grant)
                    break

        # Deduplicate: keep only the latest year for each multi-year grant
        deduplicated = deduplicate_grants(filtered_results)

        return deduplicated

    except requests.RequestException as e:
        st.error(f"Error querying NIH Reporter: {e}")
        return []


def identify_unique_pis(grants: list, search_name: str) -> dict:
    """
    Identify unique PIs from search results using profile_id.
    Groups by profile_id to properly track a person across institutions.

    Returns dict mapping profile_id -> {
        'full_name': str,
        'profile_id': str,
        'organizations': set of orgs where they have grants,
        'current_org': str (org from most recent grant),
        'primary_org': str (org with most grants),
        'grants': list of grants,
        'total_funding': float,
        'grant_count': int
    }
    """
    pi_dict = {}

    for grant in grants:
        grant_org = grant.get("organization", {}).get("org_name", "Unknown")
        grant_start = grant.get("project_start_date") or ""  # Ensure not None
        pis = grant.get("principal_investigators", [])

        for pi in pis:
            pi_full_name = pi.get("full_name", "")
            if not names_match(search_name, pi_full_name):
                continue

            profile_id = pi.get("profile_id")
            if not profile_id:
                # Fallback if no profile_id: use normalized name
                profile_id = f"name_{normalize_name(pi_full_name)}"

            if profile_id not in pi_dict:
                pi_dict[profile_id] = {
                    'full_name': pi_full_name,
                    'profile_id': profile_id,
                    'organizations': set(),
                    'org_grant_counts': {},
                    'org_latest_grant': {},  # Track most recent grant per org
                    'grants': [],
                    'total_funding': 0,
                    'grant_count': 0,
                    'latest_grant_date': "",
                    'current_org': "Unknown"
                }

            pi_dict[profile_id]['grants'].append(grant)
            pi_dict[profile_id]['organizations'].add(grant_org)
            pi_dict[profile_id]['org_grant_counts'][grant_org] = \
                pi_dict[profile_id]['org_grant_counts'].get(grant_org, 0) + 1
            pi_dict[profile_id]['total_funding'] += grant.get('award_amount', 0) or 0
            pi_dict[profile_id]['grant_count'] += 1

            # Track the most recent grant to determine current institution
            if grant_start > pi_dict[profile_id]['org_latest_grant'].get(grant_org, ""):
                pi_dict[profile_id]['org_latest_grant'][grant_org] = grant_start

            if grant_start > pi_dict[profile_id]['latest_grant_date']:
                pi_dict[profile_id]['latest_grant_date'] = grant_start
                pi_dict[profile_id]['current_org'] = grant_org

    # Determine primary organization for each PI (org with most grants)
    for profile_id, info in pi_dict.items():
        if info['org_grant_counts']:
            primary_org = max(info['org_grant_counts'].items(), key=lambda x: x[1])[0]
            info['primary_org'] = primary_org
            info['organization'] = info['current_org']  # Use current org as default
        else:
            info['primary_org'] = "Unknown"
            info['organization'] = "Unknown"
            info['current_org'] = "Unknown"

    return pi_dict


def filter_grants_for_pi(grants: list, profile_id: str) -> list:
    """
    Filter grants to only those where the specific PI (by profile_id) is listed.
    """
    filtered = []
    for grant in grants:
        pis = grant.get("principal_investigators", [])
        for pi in pis:
            if str(pi.get("profile_id", "")) == str(profile_id) or \
               (profile_id.startswith("name_") and
                normalize_name(pi.get("full_name", "")) == profile_id[5:]):
                filtered.append(grant)
                break
    return filtered


# Grant mechanisms that are ALWAYS multi-year funded (total award given upfront)
ALWAYS_MULTI_YEAR_MECHANISMS = {
    "RF1",  # Multi-year Funded Research Project Grant
    "DP1",  # NIH Director's Pioneer Award
    "DP2",  # NIH Director's New Innovator Award
    "DP5",  # NIH Director's Early Independence Award
}

# Typical annual award ranges for common mechanisms (direct costs)
# Used to detect if an award is multi-year funded based on amount
TYPICAL_ANNUAL_RANGES = {
    "R01": (150000, 800000),   # Typical R01: $150K-$800K/year
    "R21": (100000, 300000),   # R21: $100K-$300K/year
    "R35": (250000, 750000),   # MIRA: $250K-$750K/year
    "R61": (150000, 500000),   # R61: $150K-$500K/year
    "U01": (200000, 1000000),  # U01: varies widely
}


def detect_multi_year_funding(activity_code: str, award_amount: float,
                               project_start: str, project_end: str) -> bool:
    """
    Detect if a grant is multi-year funded based on award amount and project duration.

    Returns True if the award appears to be a multi-year total rather than annual.
    """
    if not activity_code or not award_amount:
        return False

    code = activity_code[:3]

    # Always multi-year funded mechanisms
    if code in ALWAYS_MULTI_YEAR_MECHANISMS:
        return True

    # Calculate project duration
    try:
        if project_start and project_end:
            start = datetime.strptime(project_start[:10], "%Y-%m-%d").date()
            end = datetime.strptime(project_end[:10], "%Y-%m-%d").date()
            total_years = (end - start).days / 365.25
        else:
            return False  # Can't determine without dates
    except:
        return False

    if total_years <= 0:
        return False

    # Calculate what the annual amount would be if this is multi-year funded
    implied_annual = award_amount / total_years

    # Get typical annual range for this mechanism
    if code in TYPICAL_ANNUAL_RANGES:
        min_annual, max_annual = TYPICAL_ANNUAL_RANGES[code]
    else:
        # Default range for unknown mechanisms
        min_annual, max_annual = 100000, 1000000

    # If award_amount is much higher than typical annual, but implied_annual is reasonable,
    # then it's likely multi-year funded
    if award_amount > max_annual * 1.5 and min_annual <= implied_annual <= max_annual * 1.2:
        return True

    return False


def calculate_remaining_funds(award_amount: float, project_start: str, project_end: str,
                              activity_code: str = "") -> tuple:
    """
    Calculate estimated remaining unspent funds for a grant.

    Automatically detects if grant is multi-year funded based on:
    1. Known multi-year mechanisms (RF1, DP1, DP2, DP5)
    2. Award amount vs project duration (if award >> typical annual but award/years is reasonable)

    For standard grants: remaining = annual_award * years_left
    For multi-year funded: remaining = total_award * (time_left / total_time)

    Returns (remaining_funds, years_remaining, total_project_years)
    """
    if not project_end or not award_amount:
        return 0.0, 0.0, 0.0

    try:
        end_date = datetime.strptime(project_end[:10], "%Y-%m-%d").date()
        today = date.today()

        if end_date <= today:
            return 0.0, 0.0, 0.0  # Grant has ended

        # Calculate years remaining
        days_remaining = (end_date - today).days
        years_remaining = days_remaining / 365.25

        # Calculate total project period
        if project_start:
            start_date = datetime.strptime(project_start[:10], "%Y-%m-%d").date()
            total_days = (end_date - start_date).days
            total_years = total_days / 365.25
        else:
            total_years = 5.0  # Assume 5-year grant if no start date

        # Detect if multi-year funded
        is_multi_year = detect_multi_year_funding(activity_code, award_amount,
                                                   project_start, project_end)

        if is_multi_year:
            # Multi-year funded: award_amount is TOTAL for entire project
            # Remaining = total * (time_remaining / total_time)
            if total_years > 0:
                fraction_remaining = years_remaining / total_years
                remaining_funds = award_amount * fraction_remaining
            else:
                remaining_funds = 0.0
        else:
            # Standard annual funding: award_amount is per year
            # Remaining = annual * years_remaining
            remaining_funds = award_amount * years_remaining

        return remaining_funds, years_remaining, total_years

    except (ValueError, TypeError):
        return 0.0, 0.0, 0.0


def calculate_recruitment_value(grants: list, target_pi_name: str) -> dict:
    """
    Calculate the estimated recruitment value from a list of grants.

    For each grant:
    - Estimates remaining unspent funds based on time remaining
    - Divides equally among all PIs for Multi-PI grants
    - Applies portability score based on grant type

    Returns a dictionary with:
    - total_value: Sum of all portable funding
    - conservative_value: Lower bound estimate
    - optimistic_value: Upper bound estimate
    - grants_detail: List of grants with individual calculations
    """
    grants_detail = []
    total_value = 0.0
    conservative_value = 0.0
    optimistic_value = 0.0
    total_remaining_unspent = 0.0

    target_name_lower = target_pi_name.lower()

    for grant in grants:
        # Extract grant info
        activity_code = grant.get("activity_code", "")
        award_amount = grant.get("award_amount", 0) or 0
        project_end = grant.get("project_end_date", "")
        project_start = grant.get("project_start_date", "")

        # Get PI information
        principal_investigators = grant.get("principal_investigators", [])
        num_pis = len(principal_investigators) if principal_investigators else 1
        is_contact_pi = False
        is_multi_pi = num_pis > 1

        for pi in principal_investigators:
            pi_name = pi.get("full_name", "").lower()
            if target_name_lower in pi_name or pi_name in target_name_lower:
                is_contact_pi = pi.get("is_contact_pi", False)
                break

        # Calculate remaining unspent funds
        remaining_funds, years_remaining, total_years = calculate_remaining_funds(
            award_amount, project_start, project_end, activity_code
        )

        # For Multi-PI: divide equally among all PIs
        if is_multi_pi:
            pi_share = 1.0 / num_pis
            pi_remaining_funds = remaining_funds * pi_share
        else:
            pi_share = 1.0
            pi_remaining_funds = remaining_funds

        # Calculate portability
        portability = get_portability_score(activity_code)

        # Calculate weighted recruitment value (portable funds)
        weighted_value = pi_remaining_funds * portability

        # Conservative estimate (additional 20% reduction for transfer friction)
        conservative = weighted_value * 0.8

        # Optimistic estimate (could retain more)
        optimistic = pi_remaining_funds * min(1.0, portability * 1.2)

        total_remaining_unspent += pi_remaining_funds
        total_value += weighted_value
        conservative_value += conservative
        optimistic_value += optimistic

        grants_detail.append({
            "project_num": grant.get("project_num", "Unknown"),
            "project_title": grant.get("project_title", "Unknown"),
            "activity_code": activity_code,
            "award_amount": award_amount,
            "project_start": project_start,
            "project_end": project_end,
            "organization": grant.get("organization", {}).get("org_name", "Unknown"),
            "portability_score": portability,
            "portability_category": get_portability_category(portability),
            "years_remaining": years_remaining,
            "total_project_years": total_years,
            "is_contact_pi": is_contact_pi,
            "is_multi_pi": is_multi_pi,
            "num_pis": num_pis,
            "pi_share": pi_share,
            "remaining_funds_total": remaining_funds,
            "pi_remaining_funds": pi_remaining_funds,
            "weighted_value": weighted_value,
            "conservative_value": conservative,
            "optimistic_value": optimistic,
        })

    # Filter out inactive grants (years_remaining <= 0)
    active_grants = [g for g in grants_detail if g["years_remaining"] > 0]

    return {
        "total_value": total_value,
        "conservative_value": conservative_value,
        "optimistic_value": optimistic_value,
        "total_remaining_unspent": total_remaining_unspent,
        "grants_detail": active_grants,
        "num_grants": len(active_grants),
    }


def format_currency(amount: float) -> str:
    """Format a number as currency."""
    if amount >= 1_000_000:
        return f"${amount/1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.1f}K"
    else:
        return f"${amount:.0f}"


# Streamlit UI
st.set_page_config(
    page_title="PI Hunter - NIH Grant Recruitment Value",
    page_icon="🔬",
    layout="wide"
)

st.title("PI Hunter")
st.subheader("NIH Grant Recruitment Value Estimator")

st.markdown("""
Search for a researcher by name to estimate the grant funding they could bring
if recruited. This tool queries NIH Reporter for active grants and calculates
a weighted recruitment value based on:
- **Grant portability** (R01s transfer easily, center grants usually don't)
- **Time remaining** on the grant
- **Multi-PI status** (contact PIs bring more of the grant)
""")

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'unique_pis' not in st.session_state:
    st.session_state.unique_pis = None
if 'selected_pi' not in st.session_state:
    st.session_state.selected_pi = None
if 'search_name' not in st.session_state:
    st.session_state.search_name = ""

# Search input
col1, col2 = st.columns([3, 1])
with col1:
    pi_name = st.text_input(
        "Researcher Name",
        placeholder="e.g., John Smith",
        help="Enter the PI's name as it appears on NIH grants"
    )
with col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

# Perform search
if search_button and pi_name:
    st.session_state.selected_pi = None  # Reset selection on new search
    st.session_state.search_name = pi_name
    with st.spinner(f"Searching NIH Reporter for '{pi_name}'..."):
        grants = search_nih_reporter(pi_name)
        st.session_state.search_results = grants

    if not grants:
        st.warning(f"No active grants found for '{pi_name}'. Try variations of the name.")
        st.session_state.unique_pis = None
    else:
        # Identify unique PIs
        unique_pis = identify_unique_pis(grants, pi_name)
        st.session_state.unique_pis = unique_pis

# Handle disambiguation if multiple PIs found
if st.session_state.unique_pis and len(st.session_state.unique_pis) > 1 and not st.session_state.selected_pi:
    st.markdown("---")
    st.subheader("Multiple researchers found")
    st.info(f"Found {len(st.session_state.unique_pis)} different researchers matching '{st.session_state.search_name}'. Please select one:")

    # Create selection options with organization info
    pi_options = []
    for profile_id, info in sorted(st.session_state.unique_pis.items(), key=lambda x: -x[1]['total_funding']):
        # Show current institution (from most recent grant) as primary identifier
        current_org = info.get('current_org', info.get('primary_org', 'Unknown'))
        orgs = info.get('organizations', set())

        if len(orgs) > 1:
            org_display = f"{current_org} (+{len(orgs)-1} other orgs)"
        else:
            org_display = current_org

        option_text = f"{info['full_name']} @ {org_display} ({info['grant_count']} grants, {format_currency(info['total_funding'])}/yr)"
        pi_options.append((profile_id, option_text, info))

    # Display as radio buttons or selectbox depending on count
    if len(pi_options) <= 10:
        selected_idx = st.radio(
            "Select researcher:",
            range(len(pi_options)),
            format_func=lambda i: pi_options[i][1],
            key="pi_selection"
        )
        if st.button("Confirm Selection", type="primary"):
            st.session_state.selected_pi = pi_options[selected_idx]
            st.rerun()
    else:
        # Too many - show as table and let user filter
        st.warning(f"Too many matches ({len(pi_options)}). Showing top results by funding:")

        # Show table of top matches
        pi_df = pd.DataFrame([
            {
                'Name': info['full_name'],
                'Current Institution': info.get('current_org', 'Unknown'),
                'Other Affiliations': ', '.join(sorted(info.get('organizations', set()) - {info.get('current_org', '')}))[:60] + ('...' if len(', '.join(sorted(info.get('organizations', set()) - {info.get('current_org', '')}))) > 60 else '') or '-',
                'Grants': info['grant_count'],
                'Annual $': format_currency(info['total_funding'])
            }
            for profile_id, info in sorted(st.session_state.unique_pis.items(), key=lambda x: -x[1]['total_funding'])[:20]
        ])
        st.dataframe(pi_df, use_container_width=True, hide_index=True)

        # Let user select from dropdown
        options_dict = {pi_options[i][1]: pi_options[i] for i in range(len(pi_options))}
        selected_option = st.selectbox(
            "Select researcher from list:",
            options=list(options_dict.keys())
        )
        if st.button("Confirm Selection", type="primary"):
            st.session_state.selected_pi = options_dict[selected_option]
            st.rerun()

# Process single PI or selected PI
grants_to_process = None
pi_name_for_calc = None
pi_info_for_display = None

if st.session_state.unique_pis:
    if len(st.session_state.unique_pis) == 1:
        # Single PI - use directly
        profile_id, info = list(st.session_state.unique_pis.items())[0]
        grants_to_process = info['grants']
        pi_name_for_calc = info['full_name']
        pi_info_for_display = info
    elif st.session_state.selected_pi:
        # User selected a PI
        profile_id, _, info = st.session_state.selected_pi
        grants_to_process = info['grants']
        pi_name_for_calc = info['full_name']
        pi_info_for_display = info

if grants_to_process:
    # Calculate recruitment value
    results = calculate_recruitment_value(grants_to_process, pi_name_for_calc)

    # Summary metrics
    st.markdown("---")
    st.subheader(f"Results for: {pi_name_for_calc}")

    # Show organization(s)
    if pi_info_for_display:
        current_org = pi_info_for_display.get('current_org', pi_info_for_display.get('primary_org', 'Unknown'))
        orgs = pi_info_for_display.get('organizations', set())
        other_orgs = orgs - {current_org}
        if len(other_orgs) > 0:
            st.caption(f"Current Institution: **{current_org}** | Also affiliated with: {', '.join(sorted(other_orgs))}")
        else:
            st.caption(f"Institution: **{current_org}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Active Grants", results["num_grants"])
    with col2:
        st.metric("Remaining Funds (PI Share)", format_currency(results["total_remaining_unspent"]),
                  help="Total estimated unspent funds for this PI's share of all grants")
    with col3:
        st.metric("Conservative", format_currency(results["conservative_value"]),
                  help="Portable value with 20% transfer friction")
    with col4:
        st.metric("Weighted Estimate", format_currency(results["total_value"]),
                  help="Remaining funds × Portability score")
    with col5:
        st.metric("Optimistic", format_currency(results["optimistic_value"]),
                  help="Best case scenario")

    # Visualization
    st.markdown("---")

    if results["grants_detail"]:
        df = pd.DataFrame(results["grants_detail"])

        # Grant breakdown chart
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("PI's Remaining Funds by Grant")
            fig = px.bar(
                df.sort_values("pi_remaining_funds", ascending=True),
                x="pi_remaining_funds",
                y="project_num",
                orientation="h",
                color="portability_category",
                color_discrete_map={
                    "Highly Portable": "#2ecc71",
                    "Likely Portable": "#3498db",
                    "Partially Portable": "#f39c12",
                    "Unlikely to Transfer": "#e74c3c"
                },
                labels={"pi_remaining_funds": "PI's Remaining Funds ($)", "project_num": "Grant"},
                hover_data=["project_title", "activity_code", "organization", "num_pis", "years_remaining"]
            )
            fig.update_layout(height=max(400, len(df) * 40))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Remaining Funds by Grant Type")
            type_summary = df.groupby("activity_code").agg({
                "pi_remaining_funds": "sum",
                "project_num": "count"
            }).reset_index()
            type_summary.columns = ["Grant Type", "PI Remaining Funds", "Count"]

            fig2 = px.pie(
                type_summary,
                values="PI Remaining Funds",
                names="Grant Type",
                hole=0.4
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # Detailed table
        st.markdown("---")
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.subheader("Grant Details")

        # Prepare export dataframe (with raw numbers for CSV)
        export_df = df[[
            "project_num", "activity_code", "project_title",
            "organization", "award_amount", "project_start", "project_end",
            "years_remaining", "num_pis", "pi_share",
            "remaining_funds_total", "pi_remaining_funds",
            "portability_score", "portability_category", "weighted_value",
            "conservative_value", "optimistic_value"
        ]].copy()

        export_df.columns = [
            "Grant Number", "Type", "Title", "Organization",
            "Annual Award ($)", "Start Date", "End Date",
            "Years Remaining", "Number of PIs", "PI Share (%)",
            "Remaining Total ($)", "Remaining PI Share ($)",
            "Portability Score", "Portability Category", "Portable Value ($)",
            "Conservative ($)", "Optimistic ($)"
        ]
        export_df["PI Share (%)"] = export_df["PI Share (%)"] * 100

        # Add summary row
        summary_row = pd.DataFrame([{
            "Grant Number": "TOTAL",
            "Type": "",
            "Title": f"Summary for {pi_name_for_calc}",
            "Organization": pi_info_for_display.get('current_org', '') if pi_info_for_display else '',
            "Annual Award ($)": df["award_amount"].sum(),
            "Start Date": "",
            "End Date": "",
            "Years Remaining": "",
            "Number of PIs": "",
            "PI Share (%)": "",
            "Remaining Total ($)": df["remaining_funds_total"].sum(),
            "Remaining PI Share ($)": results["total_remaining_unspent"],
            "Portability Score": "",
            "Portability Category": "",
            "Portable Value ($)": results["total_value"],
            "Conservative ($)": results["conservative_value"],
            "Optimistic ($)": results["optimistic_value"]
        }])
        export_df = pd.concat([export_df, summary_row], ignore_index=True)

        # CSV download button
        with col_header2:
            csv = export_df.to_csv(index=False)
            safe_name = pi_name_for_calc.replace(" ", "_").replace(",", "")
            st.download_button(
                label="Export to CSV",
                data=csv,
                file_name=f"pi_hunter_{safe_name}_{date.today().isoformat()}.csv",
                mime="text/csv",
                type="primary"
            )

        # Format display table
        display_df = df[[
            "project_num", "activity_code", "project_title",
            "organization", "award_amount", "project_end",
            "years_remaining", "num_pis", "pi_share",
            "remaining_funds_total", "pi_remaining_funds",
            "portability_category", "weighted_value"
        ]].copy()

        display_df.columns = [
            "Grant #", "Type", "Title", "Organization",
            "Annual $", "End Date", "Yrs Left", "# PIs", "PI Share",
            "Remaining (Total)", "Remaining (PI)", "Portability", "Portable Value"
        ]

        display_df["Annual $"] = display_df["Annual $"].apply(lambda x: f"${x:,.0f}")
        display_df["Yrs Left"] = display_df["Yrs Left"].apply(lambda x: f"{x:.1f}")
        display_df["PI Share"] = display_df["PI Share"].apply(lambda x: f"{x*100:.0f}%")
        display_df["Remaining (Total)"] = display_df["Remaining (Total)"].apply(lambda x: f"${x:,.0f}")
        display_df["Remaining (PI)"] = display_df["Remaining (PI)"].apply(lambda x: f"${x:,.0f}")
        display_df["Portable Value"] = display_df["Portable Value"].apply(lambda x: f"${x:,.0f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Methodology explanation
        with st.expander("Methodology & Assumptions"):
            st.markdown("""
            ### How Recruitment Value is Calculated

            **1. Remaining Unspent Funds**

            *Standard grants (R01, R21, U01, etc.):*
            ```
            Remaining = Annual Award × Years Remaining
            ```

            *Multi-year funded grants (RF1, DP1, DP2, R35):*
            ```
            Remaining = Total Award × (Years Remaining ÷ Total Project Years)
            ```
            These mechanisms provide all funding upfront in year 1.

            **2. Multi-PI Split (Equal Share)**
            - For Multi-PI grants, funds are divided equally among all PIs
            - PI Share = Remaining Funds ÷ Number of PIs
            - Example: 3 PIs on a grant → each PI gets 33.3%

            **3. Portability Score (0-1)**
            - R01, R21, R03, R37 (MERIT): 0.95 (highly portable)
            - RF1, R35 (MIRA): 0.90 (portable)
            - K-series career awards: 0.85 (follow the researcher)
            - U01 cooperative agreements: 0.50 (depends on structure)
            - P-series center grants: 0.10-0.30 (mostly institutional)
            - T-series training grants: 0.05 (stay with institution)

            **4. Final Calculation**
            ```
            Portable Value = PI's Remaining Funds × Portability Score
            ```

            **5. Deduplication**
            - Each grant counted only once (latest fiscal year kept)
            - Multi-year records (e.g., -01, -02, -03) consolidated
            - Supplements merged with parent grants

            **Estimates:**
            - **Remaining (PI Share)**: Total unspent funds attributable to this PI
            - **Conservative**: Portable value × 0.8 (accounts for transfer friction)
            - **Weighted**: Portable value (remaining × portability)
            - **Optimistic**: Remaining × (portability × 1.2)

            **Key Assumptions:**
            - Multi-PI grants are split equally (actual splits may vary)
            - Portability scores are estimates based on typical transfer patterns
            - Multi-year funded grants have funds distributed evenly over project period

            **Limitations:**
            - Does not account for pending renewals or new submissions
            - Cannot predict carryover balances or unobligated funds
            - Some grants have institutional dependencies not captured in type alone
            """)

# Footer
st.markdown("---")
st.caption(
    "Data sourced from [NIH Reporter](https://reporter.nih.gov/). "
    "This tool provides estimates only and should not be used as the sole basis for recruitment decisions. "
    "Grant transferability depends on many factors including NIH institute policies and institutional agreements."
)
