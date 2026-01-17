"""
Crawl NIH Reporter for Medical AI investigators and calculate recruitment values.

This script searches for grants related to medical AI/ML and aggregates
recruitment values by PI.
"""

import requests
import pandas as pd
from datetime import datetime, date
from collections import defaultdict
import time
import argparse

# NIH Reporter API
NIH_API_URL = "https://api.reporter.nih.gov/v2/projects/search"

# Search terms for medical AI
SEARCH_TERMS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer vision medical",
    "clinical decision support AI",
    "radiology AI",
    "pathology AI",
    "medical imaging AI",
]

# Portability scores
PORTABILITY_SCORES = {
    "R01": 0.95, "R21": 0.95, "R03": 0.95, "R15": 0.95, "R35": 0.90,
    "R33": 0.90, "R34": 0.90, "R37": 0.95, "R56": 0.90, "RF1": 0.90,
    "K01": 0.85, "K08": 0.85, "K22": 0.85, "K23": 0.85, "K25": 0.85,
    "K99": 0.95, "R00": 0.95,
    "DP1": 0.90, "DP2": 0.90, "DP5": 0.90,
    "F30": 0.95, "F31": 0.95, "F32": 0.95, "F33": 0.95,
    "U01": 0.50, "U19": 0.30, "U24": 0.25, "U54": 0.20, "UG3": 0.60, "UH3": 0.60,
    "P01": 0.30, "P20": 0.15, "P30": 0.10, "P50": 0.15,
    "T32": 0.05, "T34": 0.05, "T35": 0.05,
}

# Large grants (>$X) are almost always center/coordinating grants
# Apply a steep discount to portability since they're rarely transferable
LARGE_GRANT_PORTABILITY_DISCOUNTS = [
    (10_000_000, 0.05),  # >$10M: 5% of normal portability
    (5_000_000, 0.10),   # >$5M: 10% of normal portability
    (3_000_000, 0.25),   # >$3M: 25% of normal portability
]

ALWAYS_MULTI_YEAR = {"RF1", "DP1", "DP2", "DP5"}

TYPICAL_ANNUAL_RANGES = {
    "R01": (150000, 800000),
    "R21": (100000, 300000),
    "R35": (250000, 750000),
    "U01": (200000, 1000000),
}



def get_portability(activity_code, award_amount=0):
    """Get portability score with large grant discount."""
    if not activity_code:
        base_score = 0.50
    else:
        base_score = PORTABILITY_SCORES.get(activity_code[:3], 0.50)

    # Apply large grant discount
    for threshold, multiplier in LARGE_GRANT_PORTABILITY_DISCOUNTS:
        if award_amount >= threshold:
            return base_score * multiplier

    return base_score


def get_base_project_number(project_num):
    """Extract base project number without year suffix and app type prefix."""
    if not project_num:
        return project_num
    if '-' in project_num:
        base = project_num.rsplit('-', 1)[0]
    else:
        base = project_num
    if base and base[0].isdigit():
        base = base[1:]
    return base


def is_multi_year_funded(activity_code, award_amount, budget_start, budget_end):
    """
    Detect if grant is multi-year funded based on budget period.
    Returns (is_multi_year, budget_years) tuple.
    """
    if not activity_code:
        return False, 1.0

    code = activity_code[:3]

    # Always multi-year funded mechanisms
    if code in ALWAYS_MULTI_YEAR:
        if budget_start and budget_end:
            try:
                start = datetime.strptime(budget_start[:10], "%Y-%m-%d").date()
                end = datetime.strptime(budget_end[:10], "%Y-%m-%d").date()
                budget_years = (end - start).days / 365.25
                return True, max(1.0, budget_years)
            except (ValueError, TypeError, AttributeError):
                pass
        return True, 4.0

    # Very large grants (>$3M) are almost always multi-year funded or center grants
    if award_amount and award_amount >= 3_000_000:
        if budget_start and budget_end:
            try:
                start = datetime.strptime(budget_start[:10], "%Y-%m-%d").date()
                end = datetime.strptime(budget_end[:10], "%Y-%m-%d").date()
                budget_years = (end - start).days / 365.25
                return True, max(1.0, budget_years)
            except (ValueError, TypeError, AttributeError):
                pass
        return True, 5.0

    # Check budget period - if > 1 year, it's multi-year funded
    if budget_start and budget_end:
        try:
            start = datetime.strptime(budget_start[:10], "%Y-%m-%d").date()
            end = datetime.strptime(budget_end[:10], "%Y-%m-%d").date()
            budget_years = (end - start).days / 365.25
            if budget_years > 1.5:
                return True, budget_years
            else:
                return False, 1.0
        except (ValueError, TypeError, AttributeError):
            return False, 1.0

    return False, 1.0


def calculate_remaining(award_amount, project_start, project_end, activity_code,
                        budget_start=None, budget_end=None):
    """Calculate remaining unspent funds using budget period for multi-year detection."""
    if not project_end or not award_amount:
        return 0.0, 0.0

    try:
        project_end_date = datetime.strptime(project_end[:10], "%Y-%m-%d").date()
        today = date.today()

        if project_end_date <= today:
            return 0.0, 0.0

        # Detect if multi-year funded using budget period
        is_multi_year, budget_years = is_multi_year_funded(
            activity_code, award_amount, budget_start, budget_end
        )

        if is_multi_year and budget_start and budget_end:
            # Multi-year funded: use budget period for calculation
            budget_start_date = datetime.strptime(budget_start[:10], "%Y-%m-%d").date()
            budget_end_date = datetime.strptime(budget_end[:10], "%Y-%m-%d").date()

            if today < budget_start_date:
                years_remaining = budget_years
                remaining = award_amount
            elif today >= budget_end_date:
                return 0.0, 0.0
            else:
                days_remaining = (budget_end_date - today).days
                years_remaining = days_remaining / 365.25
                fraction_remaining = years_remaining / budget_years
                remaining = award_amount * fraction_remaining

            return remaining, years_remaining
        else:
            # Standard annual funding
            days_remaining = (project_end_date - today).days
            years_remaining = days_remaining / 365.25
            remaining = award_amount * years_remaining
            return remaining, years_remaining

    except (ValueError, TypeError, AttributeError):
        return 0.0, 0.0


def search_nih_reporter(search_term, offset=0, limit=500):
    """Search NIH Reporter for grants matching search term."""
    payload = {
        "criteria": {
            "advanced_text_search": {
                "operator": "and",
                "search_field": "all",
                "search_text": search_term
            },
            "is_active": True,
            "exclude_subprojects": True,
        },
        "offset": offset,
        "limit": limit,
        "sort_field": "award_amount",
        "sort_order": "desc"
    }

    try:
        response = requests.post(
            NIH_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", []), data.get("meta", {}).get("total", 0)
    except Exception as e:
        print(f"Error searching for '{search_term}': {e}")
        return [], 0


def deduplicate_grants(grants):
    """Keep only latest fiscal year for each base grant."""
    grant_dict = {}
    for grant in grants:
        project_num = grant.get("project_num", "")
        base_num = get_base_project_number(project_num)
        fiscal_year = grant.get("fiscal_year", 0)

        if base_num not in grant_dict:
            grant_dict[base_num] = grant
        elif fiscal_year > grant_dict[base_num].get("fiscal_year", 0):
            grant_dict[base_num] = grant

    return list(grant_dict.values())


def crawl_medical_ai_grants(max_per_term=1000, verbose=True):
    """Crawl NIH Reporter for medical AI grants."""
    all_grants = {}

    for term in SEARCH_TERMS:
        if verbose:
            print(f"Searching: '{term}'...")

        grants, total = search_nih_reporter(term, offset=0, limit=min(500, max_per_term))

        if verbose:
            print(f"  Found {total} total, retrieved {len(grants)}")

        # Store by project number to dedupe across search terms
        for g in grants:
            proj_num = g.get("project_num", "")
            if proj_num and proj_num not in all_grants:
                all_grants[proj_num] = g

        # Paginate if needed
        if total > 500 and max_per_term > 500:
            for offset in range(500, min(total, max_per_term), 500):
                time.sleep(0.5)  # Be nice to API
                more_grants, _ = search_nih_reporter(term, offset=offset, limit=500)
                for g in more_grants:
                    proj_num = g.get("project_num", "")
                    if proj_num and proj_num not in all_grants:
                        all_grants[proj_num] = g
                if verbose:
                    print(f"  Retrieved {len(all_grants)} unique grants so far...")

        time.sleep(0.3)  # Rate limiting

    grants_list = list(all_grants.values())
    if verbose:
        print(f"\nTotal unique grants: {len(grants_list)}")

    # Deduplicate multi-year records
    deduped = deduplicate_grants(grants_list)
    if verbose:
        print(f"After deduplication: {len(deduped)} grants")

    return deduped


def aggregate_by_pi(grants, verbose=True):
    """Aggregate grants by PI and calculate recruitment values."""
    pi_data = defaultdict(lambda: {
        'full_name': '',
        'profile_id': None,
        'current_org': 'Unknown',
        'organizations': set(),
        'grants': [],
        'grant_count': 0,
        'total_award': 0,
        'total_remaining': 0,
        'total_portable': 0,
        'latest_grant_date': '',
    })

    for grant in grants:
        award = grant.get("award_amount", 0) or 0
        proj_start = grant.get("project_start_date") or ""
        proj_end = grant.get("project_end_date") or ""
        budget_start = grant.get("budget_start") or ""
        budget_end = grant.get("budget_end") or ""
        activity = grant.get("activity_code", "")
        grant_org = grant.get("organization", {}).get("org_name", "Unknown")

        # Calculate remaining funds (using budget period for multi-year detection)
        remaining, years_left = calculate_remaining(
            award, proj_start, proj_end, activity, budget_start, budget_end
        )

        if remaining <= 0:
            continue  # Skip ended grants

        portability = get_portability(activity, award)

        # Process each PI on the grant
        pis = grant.get("principal_investigators") or []
        if not pis:
            continue  # Skip grants without PI info
        num_pis = len(pis)
        pi_share = 1.0 / num_pis

        for pi in pis:
            profile_id = pi.get("profile_id")
            if not profile_id:
                continue  # Skip PIs without profile_id

            pi_remaining = remaining * pi_share
            pi_portable = pi_remaining * portability

            pi_data[profile_id]['full_name'] = pi.get("full_name", "Unknown")
            pi_data[profile_id]['profile_id'] = profile_id
            pi_data[profile_id]['organizations'].add(grant_org)
            pi_data[profile_id]['grants'].append(grant.get("project_num", ""))
            pi_data[profile_id]['grant_count'] += 1
            pi_data[profile_id]['total_award'] += award * pi_share
            pi_data[profile_id]['total_remaining'] += pi_remaining
            pi_data[profile_id]['total_portable'] += pi_portable

            # Track current org (most recent grant)
            if proj_start > pi_data[profile_id]['latest_grant_date']:
                pi_data[profile_id]['latest_grant_date'] = proj_start
                pi_data[profile_id]['current_org'] = grant_org

    if verbose:
        print(f"Found {len(pi_data)} unique PIs with active grants")

    return dict(pi_data)


def create_results_table(pi_data, top_n=None):
    """Create a pandas DataFrame with results."""
    rows = []
    for profile_id, data in pi_data.items():
        rows.append({
            'PI Name': data['full_name'],
            'Profile ID': profile_id,
            'Current Institution': data['current_org'],
            'Other Institutions': ', '.join(sorted(data['organizations'] - {data['current_org']}))[:50] or '-',
            'Active Grants': data['grant_count'],
            'Remaining (PI Share)': data['total_remaining'],
            'Portable Value': data['total_portable'],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Portable Value', ascending=False)

    if top_n:
        df = df.head(top_n)

    return df


def format_currency(x):
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    elif x >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def main():
    parser = argparse.ArgumentParser(description='Crawl NIH Reporter for Medical AI investigators')
    parser.add_argument('--max-per-term', type=int, default=500,
                        help='Max grants to fetch per search term (default: 500)')
    parser.add_argument('--top', type=int, default=100,
                        help='Show top N investigators (default: 100)')
    parser.add_argument('--output', type=str, default='medical_ai_investigators.csv',
                        help='Output CSV file (default: medical_ai_investigators.csv)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 70)
    print("Medical AI Investigators - NIH Grant Recruitment Value Analysis")
    print("=" * 70)

    # Crawl grants
    grants = crawl_medical_ai_grants(max_per_term=args.max_per_term, verbose=verbose)

    # Aggregate by PI
    pi_data = aggregate_by_pi(grants, verbose=verbose)

    # Create results table
    df = create_results_table(pi_data, top_n=args.top)

    # Display top results
    print("\n" + "=" * 70)
    print(f"TOP {args.top} MEDICAL AI INVESTIGATORS BY RECRUITMENT VALUE")
    print("=" * 70)

    display_df = df.copy()
    display_df['Remaining (PI Share)'] = display_df['Remaining (PI Share)'].apply(format_currency)
    display_df['Portable Value'] = display_df['Portable Value'].apply(format_currency)

    print(display_df.to_string(index=False))

    # Save full results to CSV
    df.to_csv(args.output, index=False)
    print(f"\nFull results saved to: {args.output}")

    # Summary stats
    total_investigators = len(pi_data)
    total_portable = sum(d['total_portable'] for d in pi_data.values())
    print(f"\nSummary:")
    print(f"  Total investigators found: {total_investigators}")
    print(f"  Total portable value: {format_currency(total_portable)}")


if __name__ == "__main__":
    main()
