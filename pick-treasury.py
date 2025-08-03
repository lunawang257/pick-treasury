#! /usr/bin/env python3

import csv
import argparse
from datetime import datetime, timedelta
import calendar
from typing import Tuple, List, Dict
import statistics

# Constants
RATE_SLIPPAGE = 0.3  # Default rate slippage in percentage points
TREASURY_PERIODS = ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr']


def is_third_friday(date_str: str) -> bool:
    """
    Check if a given date is the 3rd Friday of the month.

    Args:
        date_str: Date string in format 'MM/DD/YY'

    Returns:
        bool: True if it's the 3rd Friday, False otherwise
    """
    try:
        # Parse the date
        date_obj = datetime.strptime(date_str, '%m/%d/%y')

        # Get the first day of the month
        first_day = date_obj.replace(day=1)

        # Find the first Friday
        first_friday = first_day
        while first_friday.weekday() != calendar.FRIDAY:
            first_friday += timedelta(days=1)

        # The 3rd Friday is 14 days after the first Friday
        third_friday = first_friday + timedelta(days=14)

        return date_obj.date() == third_friday.date()
    except:
        return False

def write_filtered_data(data: List[Dict], output_file: str):
    """
    Write filtered data to a tab-separated file.

    Args:
        data: Filtered treasury yield data
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', newline='') as file:
            # Define fieldnames for the output file
            fieldnames = ['Date'] + TREASURY_PERIODS
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            for row in data:
                # Create a row for writing
                output_row = {'Date': row['Date'].strftime('%Y-%m-%d')}
                for period in TREASURY_PERIODS:
                    if row[period] is not None:
                        output_row[period] = f"{row[period]:.2f}"
                    else:
                        output_row[period] = ""
                writer.writerow(output_row)

        print(f"Filtered data written to: {output_file}")
    except Exception as e:
        print(f"Error writing filtered data to {output_file}: {e}")

def load_and_filter_data(csv_file: str) -> List[Dict]:
    """
    Load CSV data and filter for 3rd Friday of each month with specified
    treasury periods. Use data range where any treasury periods are available.

    Args:
        csv_file: Path to the CSV file

    Returns:
        List[Dict]: Filtered data with only 3rd Friday data and specified
        periods
    """
    filtered_data = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            date_str = row['Date']

            # Check if it's the 3rd Friday
            if is_third_friday(date_str):
                # Parse the date
                date_obj = datetime.strptime(date_str, '%m/%d/%y')

                # Extract only the specified treasury periods
                data_point = {'Date': date_obj}
                for period in TREASURY_PERIODS:
                    try:
                        # Handle empty values
                        value = row[period]
                        if value and value.strip():
                            data_point[period] = float(value)
                        else:
                            data_point[period] = None
                    except (ValueError, KeyError):
                        data_point[period] = None

                filtered_data.append(data_point)

    # Sort by date
    filtered_data.sort(key=lambda x: x['Date'])

    # Find the range where any treasury period data
    # is available and data is complete
    start_idx = None
    end_idx = None

    # First, find where any treasury period data starts
    for i, data_point in enumerate(filtered_data):
        if any(data_point[period] is not None for period in TREASURY_PERIODS):
            start_idx = i
            break

    if start_idx is None:
        print("Warning: No treasury rate data found!")
        return []

    # Find the longest continuous range where at least some treasury
    # periods have data
    best_start = start_idx
    best_end = start_idx
    current_start = start_idx

    for i in range(start_idx, len(filtered_data)):
        data_point = filtered_data[i]

        # Check if this data point has at least some treasury period data
        has_some_data = any(
            data_point[period] is not None for period in TREASURY_PERIODS
        )

        if has_some_data:
            # Continue the current range
            current_end = i
        else:
            # End current range and start a new one
            if current_end - current_start > best_end - best_start:
                best_start = current_start
                best_end = current_end

            # Look for next valid starting point
            current_start = None
            for j in range(i + 1, len(filtered_data)):
                if any(
                    filtered_data[j][period] is not None
                    for period in TREASURY_PERIODS
                ):
                    current_start = j
                    break
            if current_start is None:
                break
            i = current_start - 1  # Adjust loop index

    # Check if the last range is the longest
    if current_end - current_start > best_end - best_start:
        best_start = current_start
        best_end = current_end

    if best_start is not None and best_end is not None:
        start_date = filtered_data[best_start]['Date'].strftime('%Y-%m-%d')
        end_date = filtered_data[best_end]['Date'].strftime('%Y-%m-%d')
        print(
            f"Using complete data range: {start_date} to {end_date}"
        )
        print(
            f"Total months with complete data: {best_end - best_start + 1}"
        )
        return filtered_data[best_start:best_end + 1]
    else:
        print("Warning: No complete data range found!")
        return []

def CmpBorrow(data: List[Dict], short_term: str, long_term: str,
              pick_threshold: float, rate_slippage: float = RATE_SLIPPAGE,
              initial_amount: float = 1000000.0,
              pick_method: str = "use_threshold",
              fixed_term: str = None, skip_terms: List[str] = None,
              suppress_warnings: bool = False) -> Dict:
    """
    Compare two money borrowing strategies using compound interest
    simulation.

    Args:
        data: List of dictionaries with treasury yield data
        short_term: Shorter term treasury period (e.g., '1 Mo', '3 Mo')
        long_term: Longer term treasury period (e.g., '3 Mo', '6 Mo')
        pick_threshold: Threshold for choosing long term vs short term
        rate_slippage: Additional borrowing cost in percentage points
        initial_amount: Initial amount to borrow (default $1M)
        pick_method: Method for choosing rates ("pick_high", "pick_low",
          "use_threshold")
        fixed_term: Fixed term to always use (e.g., "1 Mo", "3 Mo") - overrides
          other methods if specified
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        Dict: Results including compound interest cost, detailed borrowing
        history, and performance metrics
    """
    if short_term not in TREASURY_PERIODS or long_term not in TREASURY_PERIODS:
        raise ValueError(
            f"Invalid treasury periods. Must be one of {TREASURY_PERIODS}"
        )

    # Validate that short_term is actually shorter than long_term
    short_idx = TREASURY_PERIODS.index(short_term)
    long_idx = TREASURY_PERIODS.index(long_term)
    if short_idx >= long_idx:
        raise ValueError(
            f"short_term ({short_term}) must be shorter than "
            f"long_term ({long_term})"
        )

    # Define borrowing durations in months for each period
    duration_months = {
        '1 Mo': 1,
        '3 Mo': 3,
        '6 Mo': 6,
        '1 Yr': 12,
        '2 Yr': 24
    }

    # Initialize compound interest simulation
    current_amount = initial_amount
    total_interest_paid = 0.0
    borrowing_history = []

    results = {
        'total_cost': 0.0,
        'compound_interest_cost': 0.0,
        'final_amount': initial_amount,
        'borrowing_history': [],
        'decisions': [],
        'rates_used': [],
        'current_position': None,
        'position_start_date': None
    }

    i = 0
    while i < len(data):
        row = data[i]
        current_date = row['Date']
        short_rate = row[short_term]
        long_rate = row[long_term]

        # Get skipped periods for comparison
        skipped_periods = convert_skip_to_periods(skip_terms) \
            if skip_terms else []

        # Report and skip if we don't have valid rates
        # (but don't warn for skipped terms)
        if short_rate is None or long_rate is None:
            # Only warn if the missing rate is not in the skipped terms
            missing_terms = []
            if short_rate is None and short_term not in skipped_periods:
                missing_terms.append(f"{short_term}: {short_rate}")
            if long_rate is None and long_term not in skipped_periods:
                missing_terms.append(f"{long_term}: {long_rate}")

            if missing_terms and not suppress_warnings:
                print(
                    f"Warning: Missing data on"
                    f" {current_date.strftime('%Y-%m-%d')}"
                    f" - {', '.join(missing_terms)}"
                )
            i += 1
            continue

        if short_rate > 50 or long_rate > 50:
            if not suppress_warnings:
                print(
                    f"Warning: Suspiciously high rates on "
                    f"{current_date.strftime('%Y-%m-%d')} - "
                    f"{short_term}: {short_rate}, {long_term}: {long_rate}"
                )

        # Determine strategy decision based on pick_method or fixed_term
        if fixed_term is not None:
            # Fixed strategy: always use the specified term
            if fixed_term == long_term:
                strategy = long_term
                chosen_rate = long_rate
                duration = duration_months[long_term]
            elif fixed_term == short_term:
                strategy = short_term
                chosen_rate = short_rate
                duration = duration_months[short_term]
            else:
                # If fixed_term is not one of the two terms being compared,
                # use the rate from the fixed term if available
                if fixed_term in row:
                    strategy = fixed_term
                    chosen_rate = row[fixed_term]
                    duration = duration_months[fixed_term]
                else:
                    # Fall back to short term if fixed term not available
                    strategy = short_term
                    chosen_rate = short_rate
                    duration = duration_months[short_term]
        elif pick_method == "pick_high":
            # Always use the higher rate
            use_long_term = long_rate > short_rate
            if use_long_term:
                strategy = long_term
                chosen_rate = long_rate
                duration = duration_months[long_term]
            else:
                strategy = short_term
                chosen_rate = short_rate
                duration = duration_months[short_term]
        elif pick_method == "pick_low":
            # Always use the lower rate
            use_long_term = long_rate < short_rate
            if use_long_term:
                strategy = long_term
                chosen_rate = long_rate
                duration = duration_months[long_term]
            else:
                strategy = short_term
                chosen_rate = short_rate
                duration = duration_months[short_term]
        else:  # use_threshold
            if short_rate == 0:
                use_long_term = True
            else:
                # Use the original threshold-based strategy
                use_long_term = long_rate / short_rate > (1 + pick_threshold)

            if use_long_term:
                strategy = long_term
                chosen_rate = long_rate
                duration = duration_months[long_term]
            else:
                strategy = short_term
                chosen_rate = short_rate
                duration = duration_months[short_term]

        # Add rate slippage to get actual borrowing cost
        actual_rate = chosen_rate + rate_slippage

        # Calculate interest for this borrowing period
        # Convert annual rate to monthly rate, then compound for the duration
        monthly_rate = actual_rate / 12 / 100
        period_interest = (
            current_amount * monthly_rate * duration  # Prepaid
        )

        # Update compound interest simulation
        total_interest_paid += period_interest
        current_amount += period_interest  # Interest is added to the principal

        # Record detailed borrowing information
        borrowing_record = {
            'borrow_date': current_date,
            'strategy': strategy,
            'treasury_rate': chosen_rate,
            'actual_rate': actual_rate,
            'duration_months': duration,
            'principal_at_start': current_amount - period_interest,
            'interest_paid': period_interest,
            'principal_at_end': current_amount,
            'short_rate': short_rate,
            'long_rate': long_rate
        }
        borrowing_history.append(borrowing_record)

        # Update results
        results['decisions'].append({
            'date': current_date,
            'short_rate': short_rate,
            'long_rate': long_rate,
            'strategy': strategy,
            'actual_rate': actual_rate,
            'period_cost': period_interest,
            'duration': duration
        })
        results['rates_used'].append(actual_rate)

        # Update current position
        results['current_position'] = strategy
        if results['position_start_date'] is None:
            results['position_start_date'] = current_date

        # Skip ahead by the duration of the borrowing period
        i += duration

    # Calculate final results
    results['compound_interest_cost'] = total_interest_paid
    results['final_amount'] = current_amount
    results['borrowing_history'] = borrowing_history

    # Calculate annualized compound interest cost
    if borrowing_history:
        start_date = borrowing_history[0]['borrow_date']
        end_date = borrowing_history[-1]['borrow_date']
        total_years = (end_date - start_date).days / 365.25
        if total_years > 0:
            results['annualized_compound_interest_cost'] = (
                total_interest_paid / total_years)
        else:
            results['annualized_compound_interest_cost'] = total_interest_paid
    else:
        results['annualized_compound_interest_cost'] = 0.0

    # Use annualized compound interest as the cost metric
    results['total_cost'] = results['annualized_compound_interest_cost']

    # Calculate additional metrics
    if results['rates_used']:
        results['avg_rate'] = (
            sum(results['rates_used']) / len(results['rates_used'])
        )
    else:
        results['avg_rate'] = 0.0

    results['total_periods'] = len(results['decisions'])
    results['long_term_choices'] = sum(
        1 for d in results['decisions'] if d['strategy'] == long_term
    )
    results['short_term_choices'] = sum(
        1 for d in results['decisions'] if d['strategy'] == short_term
    )

    return results

def create_strategy_result(result: Dict, short_term: str, long_term: str,
                           pick_threshold: float, pick_method: str,
                           fixed_term: str = None,
                           strategy_id: str = None) -> Dict:
    """
    Create a standardized result dictionary for a strategy.

    Args:
        result: Result from CmpBorrow function
        short_term: Short term treasury period
        long_term: Long term treasury period
        pick_threshold: Threshold used
        pick_method: Method used for picking rates
        fixed_term: Fixed term used (if applicable)
        strategy_id: Unique identifier for the strategy

    Returns:
        Dict: Standardized result dictionary
    """
    # Calculate annualized rate using compound interest formula
    initial_amount = 1000000.0  # Default initial amount
    final_amount = result.get('final_amount', initial_amount)

    # Get the time period from the borrowing history
    borrowing_history = result.get('borrowing_history', [])
    if borrowing_history:
        start_date = borrowing_history[0]['borrow_date']
        end_date = borrowing_history[-1]['borrow_date']
        total_years = (end_date - start_date).days / 365.25

        if total_years > 0 and final_amount > 0:
            # Use compound interest formula:
            #   final_amount = initial_amount * (1 + rate)^years
            # Solve for rate:
            #   rate = (final_amount/initial_amount)^(1/years) - 1
            annualized_rate = (
                (final_amount / initial_amount) ** (1 / total_years) - 1
            ) * 100
        else:
            annualized_rate = 0.0
    else:
        annualized_rate = 0.0

    return {
        'strategy_id': strategy_id,
        'short_term': short_term,
        'long_term': long_term,
        'pick_threshold': pick_threshold,
        'pick_method': pick_method,
        'fixed_term': fixed_term,
        'total_cost': result['total_cost'],
        'avg_rate': result['avg_rate'],
        'annualized_compound_interest_cost':
            result.get('annualized_compound_interest_cost', 0),
        'annualized_rate': annualized_rate,
        'compound_interest_cost': result.get('compound_interest_cost', 0),
        'final_amount': result.get('final_amount', initial_amount),
        'total_periods': result['total_periods'],
        'long_term_choices': result['long_term_choices'],
        'short_term_choices': result['short_term_choices'],
        'long_term_pct': (
            result['long_term_choices'] / result['total_periods'] * 100
            if result['total_periods'] > 0 else 0
        )
    }

# Global counter for strategy IDs
_strategy_id_counter = 0

def get_next_strategy_id() -> int:
    """
    Get the next available strategy ID.

    Returns:
        int: Next strategy ID
    """
    global _strategy_id_counter
    _strategy_id_counter += 1
    return _strategy_id_counter - 1

def reset_strategy_id_counter():
    """
    Reset the strategy ID counter to 0.
    """
    global _strategy_id_counter
    _strategy_id_counter = 0

def test_strategy_combination(data: List[Dict], short_term: str, long_term: str,
                              pick_method: str,
                              pick_thresholds: List[float],
                              skip_terms: List[str] = None) -> Dict:
    """
    Test a specific combination of short and long terms with a pick method.

    Args:
        data: Treasury yield data
        short_term: Short term period
        long_term: Long term period
        pick_method: Pick method to test
        pick_thresholds: List of thresholds to test
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        Dict: Results for this combination
    """
    results = {}

    if pick_method == "use_threshold":
        # For use_threshold, test all threshold values
        for threshold in pick_thresholds:
            # Create compact strategy name
            short_compact = short_term.replace(' ', '')
            long_compact = long_term.replace(' ', '')
            if pick_method == "use_threshold":
                strategy_name = f"{short_compact}_{long_compact}" + \
                                f"_thresh_{threshold}"
            else:
                strategy_name = f"{short_compact}_{long_compact}_{pick_method}"
            strategy_id = get_next_strategy_id()

            try:
                result = CmpBorrow(
                    data, short_term, long_term, threshold,
                    pick_method=pick_method, skip_terms=skip_terms,
                    suppress_warnings=True)
                results[strategy_name] = create_strategy_result(
                    result, short_term, long_term, threshold, pick_method,
                    strategy_id=strategy_id)
            except Exception as e:
                print(f"Error testing {strategy_name}: {e}")
    else:
        # For pick_high and pick_low, threshold doesn't matter
        short_compact = short_term.replace(' ', '')
        long_compact = long_term.replace(' ', '')
        strategy_name = f"{short_compact}_{long_compact}_{pick_method}"
        strategy_id = get_next_strategy_id()

        try:
            result = CmpBorrow(data, short_term, long_term, 0.0,
                               pick_method=pick_method, skip_terms=skip_terms,
                               suppress_warnings=True)
            results[strategy_name] = create_strategy_result(
                result, short_term, long_term, 0.0, pick_method,
                strategy_id=strategy_id)
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")

    return results

def test_fixed_strategies(data: List[Dict],
                          skip_terms: List[str] = None) -> Dict:
    """
    Test fixed strategies for each treasury period.

    Args:
        data: Treasury yield data
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        Dict: Results for fixed strategies
    """
    results = {}

    # Get filtered treasury periods
    filtered_periods = get_filtered_treasury_periods(skip_terms)

    # Check which treasury periods have sufficient data
    available_periods = [period for period in filtered_periods
                        if check_data_availability(data, period)]

    if len(available_periods) < 2:
        return results

    # Use the first two available periods for comparison
    short_term = available_periods[0]
    long_term = available_periods[1]

    for term in available_periods:
        term_compact = term.replace(' ', '')
        strategy_name = f"fixed_{term_compact}"
        strategy_id = get_next_strategy_id()

        try:
            result = CmpBorrow(data, short_term, long_term, 0.0,
                               fixed_term=term, skip_terms=skip_terms,
                               suppress_warnings=True)
            results[strategy_name] = create_strategy_result(
                result, short_term, long_term, 0.0, 'fixed', fixed_term=term,
                strategy_id=strategy_id)
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")

    return results

def convert_skip_to_periods(skip_terms: List[str]) -> List[str]:
    """
    Convert skip parameters to treasury period names.

    Args:
        skip_terms: List of skip terms (e.g., ['1m', '3m'])

    Returns:
        List[str]: List of treasury period names to skip
    """
    skip_mapping = {
        '1m': '1 Mo',
        '3m': '3 Mo',
        '6m': '6 Mo',
        '1y': '1 Yr',
        '2y': '2 Yr'
    }

    return [skip_mapping.get(term.lower(), term) for term in skip_terms]

def reset_skip_info_flag():
    """
    Reset the skip info printed flag to allow printing skip information again.
    """
    if hasattr(backtest_strategies, '_skip_info_printed'):
        delattr(backtest_strategies, '_skip_info_printed')
    if hasattr(backtest_strategies, '_available_periods_printed'):
        delattr(backtest_strategies, '_available_periods_printed')
    if hasattr(find_common_data_range, '_info_printed'):
        delattr(find_common_data_range, '_info_printed')

def get_filtered_treasury_periods(skip_terms: List[str] = None) -> List[str]:
    """
    Get treasury periods after filtering out skipped terms.

    Args:
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        List[str]: Filtered list of treasury periods
    """
    if not skip_terms:
        return TREASURY_PERIODS.copy()

    skip_periods = convert_skip_to_periods(skip_terms)
    return [period for period in TREASURY_PERIODS if period not in skip_periods]

def check_data_availability(data: List[Dict], period: str) -> bool:
    """
    Check if a treasury period has sufficient data available.

    Args:
        data: List of dictionaries with treasury yield data
        period: Treasury period to check (e.g., '1 Mo', '3 Mo')

    Returns:
        bool: True if period has sufficient data, False otherwise
    """
    if not data:
        return False

    # Count how many data points have valid rates for this period
    valid_count = sum(1 for row in data if row.get(period) is not None)

    # Require at least 10% of data points to have valid rates
    min_required = max(10, len(data) * 0.1)
    return valid_count >= min_required

def find_common_data_range(data: List[Dict], required_periods: List[str]) -> List[Dict]:
    """
    Find the data range where all required treasury periods have data.

    Args:
        data: List of dictionaries with treasury yield data
        required_periods: List of treasury periods that must have data

    Returns:
        List[Dict]: Filtered data where all required periods have data
    """
    if not data or not required_periods:
        return data

    # Find data points where ALL required periods have data
    filtered_data = []
    for row in data:
        if all(row.get(period) is not None for period in required_periods):
            filtered_data.append(row)

    # Only print information once per run
    if filtered_data and not hasattr(find_common_data_range, '_info_printed'):
        start_date = filtered_data[0]['Date'].strftime('%Y-%m-%d')
        end_date = filtered_data[-1]['Date'].strftime('%Y-%m-%d')
        print(f"Common data range for all strategies: {start_date} to {end_date}")
        print(f"Total months with complete data: {len(filtered_data)}")
        find_common_data_range._info_printed = True

    return filtered_data

def backtest_strategies(
    data: List[Dict], pick_thresholds: List[float] = None,
    skip_terms: List[str] = None) -> Dict:
    """
    Backtest different treasury selection strategies.

    Args:
        data: List of dictionaries with treasury yield data
        pick_thresholds: List of pick thresholds to test
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        Dict: Results for all strategy combinations
    """
    if pick_thresholds is None:
        pick_thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]

    pick_methods = ["pick_high", "pick_low", "use_threshold"]
    results = {}

        # Get filtered treasury periods
    filtered_periods = get_filtered_treasury_periods(skip_terms)

    # Print skip information only once per run
    if skip_terms and not hasattr(backtest_strategies, '_skip_info_printed'):
        print(f"Skipping treasury periods:"
              f" {convert_skip_to_periods(skip_terms)}")
        print(f"Using treasury periods: {filtered_periods}")
        backtest_strategies._skip_info_printed = True

    # Check which treasury periods have sufficient data
    available_periods = [period for period in filtered_periods
                        if check_data_availability(data, period)]

    # Print available periods only once per run
    if not hasattr(backtest_strategies, '_available_periods_printed'):
        print(f"Available treasury periods: {available_periods}")
        backtest_strategies._available_periods_printed = True

    if len(available_periods) < 2:
        print("Warning: Need at least 2 treasury periods with data"
              " to run backtests")
        return results

    # Find common data range where all available periods have data
    common_data = find_common_data_range(data, available_periods)

    if not common_data:
        print("Warning: No common data range found for all strategies")
        return results

    # Test all combinations of short and long terms (only for available periods)
    for i, short_term in enumerate(available_periods[:-1]):
        for long_term in available_periods[i+1:]:
            for pick_method in pick_methods:
                combination_results = test_strategy_combination(
                    common_data, short_term, long_term, pick_method, pick_thresholds,
                    skip_terms)
                results.update(combination_results)

    # Test fixed strategies (only for available periods)
    fixed_results = test_fixed_strategies(common_data, skip_terms)
    results.update(fixed_results)

    return results

def find_strategy_by_id(results: Dict, strategy_id: int) -> Tuple[str, Dict]:
    """
    Find a strategy by its ID.

    Args:
        results: Results from backtest_strategies
        strategy_id: Strategy ID to find

    Returns:
        Tuple: (strategy_name, strategy_data) or (None, None) if not found
    """
    if not results:
        return None, None

    for strategy_name, strategy_data in results.items():
        if strategy_data.get('strategy_id') == strategy_id:
            return strategy_name, strategy_data

    return None, None

def find_best_strategy(results: Dict) -> Tuple[str, Dict]:
    """
    Find the best performing strategy based on total cost.

    Args:
        results: Results from backtest_strategies

    Returns:
        Tuple: (best_strategy_name, best_strategy_data)
    """
    if not results:
        return None, None

    best_strategy = min(results.items(), key=lambda x: x[1]['total_cost'])
    return best_strategy[0], best_strategy[1]

def print_results(results: Dict, top_n: int = 10, worst_n: int = 5):
    """
    Print the top N and worst N performing strategies.

    Args:
        results: Results from backtest_strategies
        top_n: Number of top strategies to display
        worst_n: Number of worst strategies to display
    """
    if not results:
        print("No results to display.")
        return

    # Sort by compound interest cost (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_cost'])

    # Print top strategies
    print(
        f"\nTop {min(top_n, len(sorted_results))} Strategies "
        f"(by annualized compound interest cost) "
        f"out of {len(sorted_results)} total strategies:"
    )
    print("-" * 130)
    print(
        f"{'Rank':<5} {'ID':<3} {'Strategy':<25}  {'Rate%':>5}  {'Long%':>5}"
    )
    print("-" * 50)

    for i, (strategy_name, data) in enumerate(sorted_results[:top_n]):
        strategy_id = data.get('strategy_id', 'N/A')
        rate = data.get('annualized_rate', 0)
        print(
            f"{i+1:<5} {strategy_id:<3} {strategy_name:<25} "
            f"{rate:>5.2f}% {data['long_term_pct']:>5.1f}%"
        )

    # Print worst strategies
    if worst_n > 0 and len(sorted_results) > top_n:
        print(
            f"\nWorst {min(worst_n, len(sorted_results) - top_n)} Strategies "
            f"(by annualized compound interest cost):"
        )
        print("-" * 130)
        print(
            f"{'Rank':<5} {'ID':<3} {'Strategy':<25} {'Rate':>5} {'Long%':>5}"
        )
        print("-" * 50)

        for i, (strategy_name, data) in enumerate(sorted_results[-worst_n:]):
            strategy_id = data.get('strategy_id', 'N/A')
            rate = data.get('annualized_rate', 0)
            rank = len(sorted_results) - worst_n + i + 1
            print(
                f"{rank:<5} {strategy_id:<3} {strategy_name:<25} "
                f"{rate:>5.2f}% {data['long_term_pct']:>5.1f}%"
            )

def print_borrowing_history(result: Dict, max_records: int = 10,
                            output_file: str = None, skip_records: int = 0):
    """
    Print detailed borrowing history for a strategy and optionally write
    to file.

    Args:
        result: Results from CmpBorrow function
        max_records: Maximum number of records to display
        output_file: Optional file path to write tab-delimited history
        skip_records: Number of records to skip from the beginning
    """
    if not result.get('borrowing_history'):
        print("No borrowing history available.")
        return

    # Skip first N records and take next max_records
    start_index = skip_records
    end_index = skip_records + max_records
    records_to_show = result['borrowing_history'][start_index:end_index]

    # Check if there are any records to show after skipping
    if not records_to_show:
        return

    # Print headers only if there are records to show
    if skip_records > 0:
        print(
            f"\nDetailed Borrowing History "
            f"(skipping first {skip_records} records,"
            f" showing next {max_records} records):"
        )
    else:
        print(
            f"\nDetailed Borrowing History "
            f"(showing first {max_records} records):"
        )
    print("-" * 140)
    print(
        "Date         Strategy  Treasury%  Actual%  Duration  Principal    "
        "Interest      New Principal"
    )
    print("-" * 140)

    # Prepare data for both console output and file writing
    history_data = []
    # Skip first N records and take next max_records
    start_index = skip_records
    end_index = skip_records + max_records
    records_to_show = result['borrowing_history'][start_index:end_index]

    # Check if there are any records to show after skipping
    if not records_to_show:
        return

    for record in records_to_show:
        date_str = record['borrow_date'].strftime('%Y-%m-%d')

        strategy = record['strategy']
        treasury_rate = record['treasury_rate']
        actual_rate = record['actual_rate']
        duration = f"{record['duration_months']}mo"
        principal = f"${record['principal_at_start']:,.0f}"
        interest = f"${record['interest_paid']:,.0f}"
        new_principal = f"${record['principal_at_end']:,.0f}"

        row_str = (
            f"{date_str:<12} {strategy:<8} {treasury_rate:<9.2f} "
            f"{actual_rate:<7.2f} {duration:<9} "
            f"{principal:<12} {interest:<12} {new_principal:<15}"
        )
        print(row_str)

        # Store data for file writing
        history_data.append({
            'Date': date_str,
            'Strategy': strategy,
            'Treasury_Rate': f"{treasury_rate:.2f}",
            'Actual_Rate': f"{actual_rate:.2f}",
            'Duration': f"{record['duration_months']}mo",
            'Principal': f"{record['principal_at_start']:,.0f}",
            'Interest': f"{record['interest_paid']:,.0f}",
            'New_Principal': f"{record['principal_at_end']:,.0f}"
        })

    total_records = len(result['borrowing_history'])
    shown_records = len(records_to_show)
    remaining_after_skip = total_records - skip_records - shown_records

    if remaining_after_skip > 0:
        print(f"... and {remaining_after_skip} more records")

    # Write to file if specified
    if output_file:
        try:
            with open(output_file, 'w', newline='') as file:
                # Write header
                fieldnames = ['Date', 'Strategy', 'Treasury_Rate',
                              'Actual_Rate', 'Duration',
                              'Principal', 'Interest', 'New_Principal']
                writer = csv.DictWriter(file, fieldnames=fieldnames,
                                        delimiter='\t')
                writer.writeheader()

                # Write all records
                for record in result['borrowing_history']:
                    date_str = record['borrow_date'].strftime('%Y-%m-%d')
                    writer.writerow({
                        'Date': date_str,
                        'Strategy': record['strategy'],
                        'Treasury_Rate': f"{record['treasury_rate']:.2f}",
                        'Actual_Rate': f"{record['actual_rate']:.2f}",
                        'Duration': f"{record['duration_months']}mo",
                        'Principal': f"{record['principal_at_start']:,.0f}",
                        'Interest': f"{record['interest_paid']:,.0f}",
                        'New_Principal':
                             f"{record['principal_at_end']:,.0f}"
                    })
            print(f"\nBorrowing history written to: {output_file}")
        except Exception as e:
            print(f"Error writing to file {output_file}: {e}")

def print_strategy_summary(result: Dict, strategy_name: str = ""):
    """
    Print a summary of strategy results.

    Args:
        result: Results from CmpBorrow function
        strategy_name: Name of the strategy (optional)
    """
    long_term_pct = result['long_term_choices'] / result['total_periods'] * 100
    prefix = f"{strategy_name}: " if strategy_name else ""

    # Calculate annualized rate using compound interest formula
    initial_amount = 1000000.0  # Default initial amount
    final_amount = result.get('final_amount', initial_amount)

    # Get the time period from the borrowing history
    borrowing_history = result.get('borrowing_history', [])
    if borrowing_history:
        start_date = borrowing_history[0]['borrow_date']
        end_date = borrowing_history[-1]['borrow_date']
        total_years = (end_date - start_date).days / 365.25

        if total_years > 0 and final_amount > 0:
            # Use compound interest formula:
            #   final_amount = initial_amount * (1 + rate)^years
            # Solve for rate:
            #   rate = (final_amount/initial_amount)^(1/years) - 1
            rate = (
                (final_amount / initial_amount) ** (1 / total_years) - 1
            ) * 100
        else:
            rate = 0.0
    else:
        rate = 0.0

    print(
        f"{prefix}Annualized Cost = ${result['total_cost']:,.0f}, "
        f"Total Interest = ${result.get('compound_interest_cost', 0):,.0f}, "
        f"Rate = {rate:,.2f}%, "
        f"Long Term % = {long_term_pct:.1f}%"
    )

def run_single_strategy(data: List[Dict], short_term: str, long_term: str,
                        threshold: float, pick_method: str,
                        fixed_term: str = None, skip_terms: List[str] = None,
                        suppress_warnings: bool = False) -> Dict:
    """
    Run a single strategy and return results.

    Args:
        data: Treasury yield data
        short_term: Short term period
        long_term: Long term period
        threshold: Threshold value
        pick_method: Pick method to use
        fixed_term: Fixed term to use (if applicable)
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        Dict: Strategy results
    """
    try:
        return CmpBorrow(data, short_term, long_term, threshold,
                         pick_method=pick_method, fixed_term=fixed_term,
                         skip_terms=skip_terms,
                         suppress_warnings=suppress_warnings)
    except Exception as e:
        print(f"Error with {pick_method} (threshold {threshold}): {e}")
        return None

def generate_strategy_history(data: List[Dict], strategy_name: str,
                              strategy_data: Dict, skip_terms: List[str] = None,
                              suppress_warnings: bool = False) -> Dict:
    """
    Generate detailed borrowing history for a specific strategy.

    Args:
        data: Treasury yield data
        strategy_name: Name of the strategy
        strategy_data: Strategy data containing parameters
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        Dict: Detailed borrowing history results
    """
    try:
        short_term = strategy_data['short_term']
        long_term = strategy_data['long_term']
        pick_threshold = strategy_data['pick_threshold']
        pick_method = strategy_data['pick_method']
        fixed_term = strategy_data.get('fixed_term')

        result = CmpBorrow(data, short_term, long_term, pick_threshold,
                           pick_method=pick_method, fixed_term=fixed_term,
                           skip_terms=skip_terms,
                           suppress_warnings=suppress_warnings)

        return result
    except Exception as e:
        print(f"Error generating history for strategy {strategy_name}: {e}")
        return None

def write_strategy_history(data: List[Dict], strategy_name: str,
                           strategy_data: Dict,
                           strategy_id: str, skip_records: int = 0,
                           skip_terms: List[str] = None,
                           suppress_warnings: bool = True) -> bool:
    """
    Generate and write detailed borrowing history for a strategy to a CSV file.

    Args:
        data: Treasury yield data
        strategy_name: Name of the strategy
        strategy_data: Strategy configuration data
        strategy_id: Strategy ID for filename
        skip_records: Number of records to skip from the beginning
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        bool: True if successful, False otherwise
    """
    #print(f"\nProcessing Strategy ID {strategy_id}: {strategy_name}")

    # Generate detailed history
    history_result = generate_strategy_history(data, strategy_name,
                                               strategy_data, skip_terms,
                                               suppress_warnings)
    if history_result:
        # Create output filename with strategy name
        output_filename = f"results/borrow-history" + \
                          f"-{strategy_id}-{strategy_name}.csv"
        print_borrowing_history(
            history_result,
            max_records=len(history_result['borrowing_history']),
            output_file=output_filename, skip_records=skip_records)
        return True
    else:
        print(f"  Failed to generate history for strategy {strategy_name}")
        return False

def test_strategy_stability(data: List[Dict], n_years: int = 5,
                            skip_terms: List[str] = None) -> Dict:
    """
    Test the stability of each strategy by running backtests on rolling N-year
    periods.

    Args:
        data: List of dictionaries with treasury yield data
        n_years: Number of years for each rolling period (default: 5)
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])

    Returns:
        Dict: Stability results with mean and standard deviation of ranks for
        each strategy
    """
    if len(data) < n_years * 12:  # Need at least N years of monthly data
        print(f"Warning: Not enough data for {n_years}-year stability test. "
              f"Need at least {n_years * 12} months, but only have"
              f" {len(data)} months.")
        return {}

    # Calculate number of rolling periods
    months_per_period = n_years * 12
    num_periods = len(data) - months_per_period + 1

    print(f"Testing strategy stability over"
          f" {num_periods} rolling {n_years}-year periods...")

    # Store ranks for each strategy across all periods
    strategy_ranks = {}

    for period_idx in range(num_periods):
        # Extract data for this period
        start_idx = period_idx
        end_idx = period_idx + months_per_period
        period_data = data[start_idx:end_idx]

        # Get start and end dates for this period
        start_date = period_data[0]['Date']
        end_date = period_data[-1]['Date']

                # Only print period info for first 3 periods
        if period_idx < 3:
            print(f"Period {period_idx + 1}/{num_periods}:"
                  f" {start_date.strftime('%Y-%m')} to"
                  f" {end_date.strftime('%Y-%m')}")

        # Run backtest for this period
        try:
            period_results = backtest_strategies(
                period_data, skip_terms=skip_terms)

            if not period_results:
                if period_idx < 3:
                    print(f"  No results for period {period_idx + 1}")
                continue

            if period_idx < 3:
                print(f"  Found {len(period_results)} strategies for"
                      f" period {period_idx + 1}")

            # Sort strategies by total_cost (lower is better) and assign ranks
            sorted_strategies = sorted(period_results.items(),
                                       key=lambda x: x[1]['total_cost'])

            # Assign ranks (1 is best, higher numbers are worse)
            for rank, (strategy_name, strategy_data) in \
                enumerate(sorted_strategies, 1):
                strategy_id = strategy_data.get('strategy_id', 'unknown')

                # Use strategy name as key instead of ID since IDs change
                # between periods
                if strategy_name not in strategy_ranks:
                    strategy_ranks[strategy_name] = {
                        'name': strategy_name,
                        'ranks': [],
                        'short_term': strategy_data.get('short_term', ''),
                        'long_term': strategy_data.get('long_term', ''),
                        'pick_method': strategy_data.get('pick_method', ''),
                        'pick_threshold':
                            strategy_data.get('pick_threshold', 0.0),
                        'fixed_term': strategy_data.get('fixed_term', '')
                    }

                strategy_ranks[strategy_name]['ranks'].append(rank)

            # Debug: print first few strategies for this period
            if period_idx < 3:  # Only for first few periods
                print(f"    Top 3 strategies for period {period_idx + 1}:")
                for i, (name, strategy_data) in \
                    enumerate(sorted_strategies[:3]):
                    print(f"      {i+1}. {name} (ID:"
                          f" {strategy_data.get('strategy_id', 'N/A')})")

        except Exception as e:
            print(f"  Error in period {period_idx + 1}: {e}")
            continue

    # Calculate statistics for each strategy
    stability_results = {}

    for strategy_name, strategy_info in strategy_ranks.items():
        if len(strategy_info['ranks']) < 2:
            continue  # Skip strategies with insufficient data

        ranks = strategy_info['ranks']
        mean_rank = statistics.mean(ranks)
        stdev_rank = statistics.stdev(ranks) if len(ranks) > 1 else 0

        stability_results[strategy_name] = {
            'name': strategy_info['name'],
            'short_term': strategy_info['short_term'],
            'long_term': strategy_info['long_term'],
            'pick_method': strategy_info['pick_method'],
            'pick_threshold': strategy_info['pick_threshold'],
            'fixed_term': strategy_info['fixed_term'],
            'mean_rank': mean_rank,
            'stdev_rank': stdev_rank,
            'num_periods': len(ranks),
            'min_rank': min(ranks),
            'max_rank': max(ranks)
        }

    return stability_results

def print_stability_results(stability_results: Dict):
    """
    Print stability results sorted by mean rank.

    Args:
        stability_results: Results from test_strategy_stability
    """
    if not stability_results:
        print("No stability results to display.")
        return

    # Sort by mean rank (lower is better)
    sorted_results = sorted(stability_results.items(),
                            key=lambda x: x[1]['mean_rank'])

    print(f"\nStrategy Stability Results (sorted by mean rank):")
    print("=" * 120)
    print(f"{'Strategy Name':<50} {'Mean':<6} {'StDev':<6} {'Min':<4}"
          f" {'Max':<4} {'Periods':<8}")
    print("-" * 120)

    for strategy_name, strategy_data in sorted_results:
        mean_rank = strategy_data['mean_rank']
        stdev_rank = strategy_data['stdev_rank']
        min_rank = strategy_data['min_rank']
        max_rank = strategy_data['max_rank']
        num_periods = strategy_data['num_periods']

        print(f"{strategy_name:<50} {mean_rank:<6.1f} {stdev_rank:<6.1f}"
              f" {min_rank:<4} {max_rank:<4} {num_periods:<8}")

    print("-" * 120)
    print(f"Total strategies analyzed: {len(sorted_results)}")

def test_comprehensive_stability(data: List[Dict],
                                 skip_terms: List[str] = None,
                                 no_overlap: bool = False,
                                 use_new_data: bool = False,
                                 weighted: bool = False) -> Dict:
    """
    Test strategy stability across all possible year lengths and aggregate
    results.

    Args:
        data: List of dictionaries with treasury yield data
        skip_terms: List of terms to skip (e.g., ['1m', '3m'])
        no_overlap: If True, use non-overlapping periods instead of rolling periods
        use_new_data: If True and no_overlap is True, skip old data and use latest data
        weighted: If True, calculate weighted average and stdev using year length as weight

    Returns:
        Dict: Comprehensive stability results with aggregated statistics
    """
    period_type = "non-overlapping" if no_overlap else "rolling"
    data_strategy = "latest" if use_new_data and no_overlap else "earliest"
    weight_type = "weighted" if weighted else "unweighted"
    print(f"Running comprehensive stability test across all possible"
          f" year lengths using {period_type} periods with {data_strategy} data"
          f" ({weight_type} calculations)...")

    # Calculate maximum possible years (need at least 12 months per period)
    max_years = len(data) // 12
    print(f"Data spans {len(data)} months, testing year lengths from 4 to"
          f" {max_years}")

    # Store all ranks for each strategy across all year lengths
    strategy_all_ranks = {}
    strategy_all_weights = {}

    for n_years in range(4, max_years + 1):
        print(f"\nTesting {n_years}-year periods...")

        # Calculate number of periods for this year length
        months_per_period = n_years * 12

        if no_overlap:
            # Use non-overlapping periods
            num_periods = len(data) // months_per_period
        else:
            # Use rolling periods (original behavior)
            num_periods = len(data) - months_per_period + 1

        if num_periods < 2:
            print(f"  Skipping {n_years}-year periods"
                  f" (only {num_periods} periods available)")
            continue

        period_type_str = "non-overlapping" if no_overlap else "rolling"
        print(f"  Running {num_periods} {period_type_str} {n_years}-year periods...")

                # Store ranks for this year length
        strategy_ranks = {}

        for period_idx in range(num_periods):
            # Extract data for this period
            if no_overlap:
                if use_new_data:
                    # Use latest data: skip old data and use the most recent periods
                    total_used_months = num_periods * months_per_period
                    start_offset = len(data) - total_used_months
                    start_idx = start_offset + (period_idx * months_per_period)
                    end_idx = start_idx + months_per_period
                else:
                    # Use earliest data: start from the beginning
                    start_idx = period_idx * months_per_period
                    end_idx = start_idx + months_per_period
            else:
                # Rolling periods: each period starts one month after the previous
                start_idx = period_idx
                end_idx = period_idx + months_per_period

            period_data = data[start_idx:end_idx]

            # Run backtest for this period
            try:
                period_results = backtest_strategies(period_data,
                                                     skip_terms=skip_terms)

                if not period_results:
                    continue

                # Sort strategies by total_cost (lower is better) and
                # assign ranks
                sorted_strategies = sorted(period_results.items(),
                                           key=lambda x: x[1]['total_cost'])

                # Assign ranks (1 is best, higher numbers are worse)
                for rank, (strategy_name, strategy_data) in \
                    enumerate(sorted_strategies, 1):
                    if strategy_name not in strategy_ranks:
                        strategy_ranks[strategy_name] = []

                    strategy_ranks[strategy_name].append(rank)

            except Exception as e:
                print(f"    Error in period {period_idx + 1}: {e}")
                continue

        # Aggregate results for this year length
        for strategy_name, ranks in strategy_ranks.items():
            if len(ranks) < 2:
                continue  # Skip strategies with insufficient data

            if strategy_name not in strategy_all_ranks:
                strategy_all_ranks[strategy_name] = []
                strategy_all_weights[strategy_name] = []

            # Add all ranks from this year length to the overall collection
            strategy_all_ranks[strategy_name].extend(ranks)
            # Add weights (number of years) for each rank
            strategy_all_weights[strategy_name].extend([n_years] * len(ranks))

    # Calculate comprehensive statistics for each strategy
    comprehensive_results = {}

    for strategy_name, all_ranks in strategy_all_ranks.items():
        if len(all_ranks) < 10:  # Require at least 10 total periods
            continue

        if weighted:
            # Calculate weighted statistics
            weights = strategy_all_weights[strategy_name]

            # Weighted mean
            weighted_sum = sum(rank * weight for rank, weight in zip(all_ranks, weights))
            total_weight = sum(weights)
            mean_rank = weighted_sum / total_weight if total_weight > 0 else 0

            # Weighted standard deviation
            if len(all_ranks) > 1:
                weighted_variance = sum(weight * (rank - mean_rank) ** 2
                                     for rank, weight in zip(all_ranks, weights))
                stdev_rank = (weighted_variance / total_weight) ** 0.5
            else:
                stdev_rank = 0
        else:
            # Calculate unweighted statistics
            mean_rank = statistics.mean(all_ranks)
            stdev_rank = statistics.stdev(all_ranks) if len(all_ranks) > 1 else 0

        comprehensive_results[strategy_name] = {
            'name': strategy_name,
            'mean_rank': mean_rank,
            'stdev_rank': stdev_rank,
            'num_periods': len(all_ranks),
            'min_rank': min(all_ranks),
            'max_rank': max(all_ranks),
            'total_tests': len(all_ranks)
        }

    return comprehensive_results

def print_comprehensive_stability_results(comprehensive_results: Dict):
    """
    Print comprehensive stability results sorted by mean rank.

    Args:
        comprehensive_results: Results from test_comprehensive_stability
    """
    if not comprehensive_results:
        print("No comprehensive stability results to display.")
        return

    # Sort by mean rank (lower is better)
    sorted_results = sorted(comprehensive_results.items(),
                            key=lambda x: x[1]['mean_rank'])

    print(f"\nComprehensive Strategy Stability Results"
          f" (all year lengths, sorted by mean rank):")
    print("=" * 130)
    print(f"{'Strategy Name':<50} {'Mean':<6} {'StDev':<6} {'Min':<4}"
          f" {'Max':<4} {'Total':<6}")
    print("-" * 130)

    for strategy_name, strategy_data in sorted_results:
        mean_rank = strategy_data['mean_rank']
        stdev_rank = strategy_data['stdev_rank']
        min_rank = strategy_data['min_rank']
        max_rank = strategy_data['max_rank']
        total_tests = strategy_data['total_tests']

        print(f"{strategy_name:<50} {mean_rank:<6.1f} {stdev_rank:<6.1f}"
              f" {min_rank:<4} {max_rank:<4} {total_tests:<6}")

    print("-" * 130)
    print(f"Total strategies analyzed: {len(sorted_results)}")
    print(f"Results aggregated across all year lengths (4 to maximum possible)")

def main():
    """
    Main function to run the treasury backtesting analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Treasury Bill Selection Strategy Backtester'
    )
    parser.add_argument(
        '--output-file', '-o',
        default='borrow-history.csv',
        help='Output file for borrowing history (default: borrow_history.txt)'
    )
    parser.add_argument(
        '--filtered-data-file', '-f',
        default='monthly-yield.csv',
        help='Output file for filtered monthly yield data' + \
             '(default: monthly-yield.csv)'
    )
    parser.add_argument(
        '--strategy-id', '-s',
        type=int,
        help='Generate detailed borrowing history for specific strategy ID' + \
            ' (use -1 for all strategies)'
    )
    parser.add_argument(
        '--skip-records', '-k',
        type=int,
        default=1000000,
        help='Number of records to skip from the beginning when printing '
             'detailed history (default: 1000000)'
    )
    parser.add_argument(
        '--stability-test', '-t',
        action='store_true',
        help='Run strategy stability test with rolling N-year periods'
    )
    parser.add_argument(
        '--stability-years', '-y',
        type=int,
        default=5,
        help='Number of years for stability test rolling periods (default: 5)'
    )
    parser.add_argument(
        '--comprehensive-stability', '-c',
        action='store_true',
        help='Run comprehensive stability test across all possible year lengths'
    )
    parser.add_argument(
        '--no-overlap',
        action='store_true',
        help='Use non-overlapping periods for comprehensive stability test'
    )
    parser.add_argument(
        '--use-new-data',
        action='store_true',
        help='When using --no-overlap, skip old data and use latest data'
    )
    parser.add_argument(
        '--weighted',
        action='store_true',
        help='Calculate weighted average and stdev for comprehensive stability test'
    )
    parser.add_argument(
        '--skip', '-x',
        action='append',
        choices=['1m', '3m', '6m', '1y', '2y'],
        help='Skip specific treasury terms (can be used multiple times,'
             ' e.g., --skip 1m --skip 3m)'
    )
    args = parser.parse_args()

    print("Treasury Bill Selection Strategy Backtester")
    print("=" * 50)

    # Reset skip info flag at the beginning of each run
    reset_skip_info_flag()

    # Load and filter data
    print("Loading and filtering data...")
    try:
        data = load_and_filter_data("yield-curve-rates-1990-2024.csv")
        print(f"Loaded {len(data)} data points (3rd Friday of each month)")

        if data:
            print(f"Date range: {data[0]['Date']} to {data[-1]['Date']}")

            # Write filtered data to file
            write_filtered_data(data, args.filtered_data_file)

            # Display sample of filtered data
            print("\nSample of filtered data:")
            print("Date\t\t1 Mo\t3 Mo\t6 Mo\t1 Yr\t2 Yr")
            print("-" * 50)
            for i, row in enumerate(data[:5]):
                date_str = row['Date'].strftime('%Y-%m-%d')
                mo1 = (
                    f"{row['1 Mo']:.2f}" if row['1 Mo'] is not None else "N/A"
                )
                mo3 = (
                    f"{row['3 Mo']:.2f}" if row['3 Mo'] is not None else "N/A"
                )
                mo6 = (
                    f"{row['6 Mo']:.2f}" if row['6 Mo'] is not None else "N/A"
                )
                yr1 = (
                    f"{row['1 Yr']:.2f}" if row['1 Yr'] is not None else "N/A"
                )
                yr2 = (
                    f"{row['2 Yr']:.2f}" if row['2 Yr'] is not None else "N/A"
                )
                print(f"{date_str}\t{mo1}\t{mo3}\t{mo6}\t{yr1}\t{yr2}")
        else:
            print("No data found!")
            return

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Run backtest
    print("\nRunning backtest...")
    results = backtest_strategies(data, skip_terms=args.skip)

    # Get the common data range used by backtest_strategies
    filtered_periods = get_filtered_treasury_periods(args.skip)
    available_periods = [period for period in filtered_periods
                        if check_data_availability(data, period)]
    common_data = find_common_data_range(data, available_periods) if available_periods else data

    # Find and display best strategy
    best_strategy_name, best_strategy_data = find_best_strategy(results)

    if best_strategy_data:
        print(f"\nBest Strategy: {best_strategy_name}")
        print(f"Short Term: {best_strategy_data['short_term']}")
        print(f"Long Term: {best_strategy_data['long_term']}")
        print(f"Pick Method: " + \
              f"{best_strategy_data.get('pick_method', 'use_threshold')}")
        print(f"Pick Threshold: {best_strategy_data['pick_threshold']}")
        print(
            f"Annualized Interest Cost: "
            f"${best_strategy_data['total_cost']:,.0f}"
        )
        print(
            f"Total Interest Paid: "
            f"${best_strategy_data.get('compound_interest_cost', 0):,.0f}"
        )
        print(
            f"Final Amount: "
            f"${best_strategy_data.get('final_amount', 0):,.0f}"
        )
        print(f"Average Rate: {best_strategy_data['avg_rate']:.2f}%")
        print(
            f"Long Term Choices: {best_strategy_data['long_term_choices']} "
            f"({best_strategy_data['long_term_pct']:.1f}%)"
        )
        print(
            f"Short Term Choices: {best_strategy_data['short_term_choices']} "
            f"({100-best_strategy_data['long_term_pct']:.1f}%)"
        )

        # Show detailed borrowing history for the best strategy
        try:
            best_result = run_single_strategy(
                common_data, best_strategy_data['short_term'],
                best_strategy_data['long_term'],
                best_strategy_data['pick_threshold'],
                best_strategy_data.get('pick_method', 'use_threshold'),
                skip_terms=args.skip,
                suppress_warnings=True
            )
            if best_result:
                print_borrowing_history(best_result, max_records=10,
                                        output_file=args.output_file,
                                        skip_records=args.skip_records)
        except Exception as e:
            print(f"Could not display borrowing history: {e}")

    # Print top and worst strategies
    print_results(results, top_n=len(results), worst_n=0)

        # Handle strategy ID parameter
    if args.strategy_id is not None:
        if args.strategy_id == -1:
            # Generate history for all strategies
            print(f"\n" + "=" * 50)
            print(f"Generating detailed borrowing history for ALL strategies")
            print("=" * 50)

            for strategy_name, strategy_data in results.items():
                strategy_id = strategy_data.get('strategy_id', 'N/A')
                write_strategy_history(common_data, strategy_name, strategy_data,
                                       strategy_id, 0, args.skip,
                                       suppress_warnings=True)
        else:
            # Generate history for specific strategy ID
            strategy_name, strategy_data = find_strategy_by_id(
                results, args.strategy_id)
            if strategy_name and strategy_data:
                print(f"\n" + "=" * 50)
                print(f"Generating detailed borrowing history' + \
                      f' for Strategy ID {args.strategy_id}")
                print(f"Strategy: {strategy_name}")
                print("=" * 50)

                write_strategy_history(common_data, strategy_name, strategy_data,
                                       str(args.strategy_id), 0, args.skip,
                                       suppress_warnings=True)
            else:
                print(f"Strategy ID {args.strategy_id} not found."
                      f" Available strategy IDs:")
                for strategy_name, strategy_data in results.items():
                    strategy_id = strategy_data.get('strategy_id', 'N/A')
                    print(f"  ID {strategy_id}: {strategy_name}")

    # Example of specific strategy comparison
    print("\n" + "=" * 50)
    print("Example: Comparing 1 Mo vs 3 Mo with different pick methods")
    print("=" * 50)

    # Test pick_high and pick_low
    for pick_method in ["pick_high", "pick_low"]:
        result = run_single_strategy(common_data, '1 Mo', '3 Mo', 0.0, pick_method,
                                     skip_terms=args.skip,
                                     suppress_warnings=True)
        if result:
            print_strategy_summary(result, pick_method)

    # Test use_threshold with different thresholds
    print("\nThreshold-based strategies:")
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.5]:
        result = run_single_strategy(common_data, '1 Mo', '3 Mo', threshold,
                                     "use_threshold", skip_terms=args.skip,
                                     suppress_warnings=True)
        if result:
            print_strategy_summary(
                result, f"use_threshold (threshold {threshold})")

    # Test fixed strategies
    print("\nFixed strategies:")
    for term in TREASURY_PERIODS:
        result = run_single_strategy(common_data, '1 Mo', '3 Mo', 0.0,
                                     'fixed', fixed_term=term,
                                     skip_terms=args.skip,
                                     suppress_warnings=True)
        if result:
            print_strategy_summary(result, f"fixed_{term}")

    # Test strategy stability if requested
    if args.stability_test:
        print("\n" + "=" * 50)
        print(f"Testing Strategy Stability"
              f" ({args.stability_years}-year rolling periods)...")
        print("=" * 50)
        stability_results = test_strategy_stability(
            data, n_years=args.stability_years, skip_terms=args.skip)
        print_stability_results(stability_results)

    # Test comprehensive stability if requested
    if args.comprehensive_stability:
        print("\n" + "=" * 50)
        period_type = "non-overlapping" if args.no_overlap else "rolling"
        data_strategy = "latest" if args.use_new_data and args.no_overlap else "earliest"
        weight_type = "weighted" if args.weighted else "unweighted"
        print(f"Testing Comprehensive Strategy Stability"
              f" across all year lengths using {period_type} periods with {data_strategy} data"
              f" ({weight_type} calculations)...")
        print("=" * 50)
        comprehensive_results = test_comprehensive_stability(
            data, skip_terms=args.skip, no_overlap=args.no_overlap,
            use_new_data=args.use_new_data, weighted=args.weighted)
        print_comprehensive_stability_results(comprehensive_results)

if __name__ == "__main__":
    main()
