import csv
from datetime import datetime, timedelta
import calendar
from typing import Tuple, List, Dict

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

def load_and_filter_data(csv_file: str) -> List[Dict]:
    """
    Load CSV data and filter for 3rd Friday of each month with specified
    treasury periods. Only use data range where 1-month rates are available.

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

    # Find the range where 1-month rates are available and data is complete
    start_idx = None
    end_idx = None

    # First, find where 1-month data starts
    for i, data_point in enumerate(filtered_data):
        if data_point['1 Mo'] is not None:
            start_idx = i
            break

    if start_idx is None:
        print("Warning: No 1-month rate data found!")
        return []

    # Find the longest continuous range where all required periods have data
    best_start = start_idx
    best_end = start_idx
    current_start = start_idx

    for i in range(start_idx, len(filtered_data)):
        data_point = filtered_data[i]

        # Check if this data point has all required periods
        has_all_data = all(
            data_point[period] is not None for period in TREASURY_PERIODS
        )

        if has_all_data:
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
                if all(
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
              initial_amount: float = 1000000.0) -> Dict:
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

        # Report and skip if we don't have valid rates
        if short_rate is None or long_rate is None:
            print(
                f"Warning: Missing data on {current_date.strftime('%Y-%m-%d')}"
                f" - {short_term}: {short_rate}, {long_term}: {long_rate}"
            )
            i += 1
            continue

        if short_rate > 50 or long_rate > 50:
            print(
                f"Warning: Suspiciously high rates on "
                f"{current_date.strftime('%Y-%m-%d')} - "
                f"{short_term}: {short_rate}, {long_term}: {long_rate}"
            )

        # Calculate the rate ratio
        rate_ratio = long_rate / short_rate if short_rate > 0 else float('inf')

        # Determine strategy decision
        if rate_ratio > (1 + pick_threshold):
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
            'rate_ratio': rate_ratio,
            'short_rate': short_rate,
            'long_rate': long_rate
        }
        borrowing_history.append(borrowing_record)

        # Update results
        results['decisions'].append({
            'date': current_date,
            'short_rate': short_rate,
            'long_rate': long_rate,
            'rate_ratio': rate_ratio,
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

def backtest_strategies(
    data: List[Dict], pick_thresholds: List[float] = None
) -> Dict:
    """
    Backtest different treasury selection strategies.

    Args:
        data: List of dictionaries with treasury yield data
        pick_thresholds: List of pick thresholds to test

    Returns:
        Dict: Results for all strategy combinations
    """
    if pick_thresholds is None:
        pick_thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]

    results = {}

    # Test all combinations of short and long terms
    for i, short_term in enumerate(TREASURY_PERIODS[:-1]):
        for long_term in TREASURY_PERIODS[i+1:]:
            for threshold in pick_thresholds:
                strategy_name = (
                    f"{short_term}_vs_{long_term}_thresh_{threshold}"
                )

                try:
                    result = CmpBorrow(data, short_term, long_term, threshold)
                    results[strategy_name] = {
                        'short_term': short_term,
                        'long_term': long_term,
                        'pick_threshold': threshold,
                        'total_cost': result['total_cost'],
                        'avg_rate': result['avg_rate'],
                        'total_periods': result['total_periods'],
                        'long_term_choices': result['long_term_choices'],
                        'short_term_choices': result['short_term_choices'],
                        'long_term_pct': (
                            result['long_term_choices'] /
                            result['total_periods'] * 100
                            if result['total_periods'] > 0 else 0
                        )
                    }
                except Exception as e:
                    print(f"Error testing {strategy_name}: {e}")

    return results

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

def print_results(results: Dict, top_n: int = 10):
    """
    Print the top N performing strategies.

    Args:
        results: Results from backtest_strategies
        top_n: Number of top strategies to display
    """
    if not results:
        print("No results to display.")
        return

    # Sort by compound interest cost (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_cost'])

    print(
        f"\nTop {min(top_n, len(sorted_results))} Strategies "
        f"(by annualized compound interest cost) "
        f"out of {len(sorted_results)} total strategies:"
    )
    print("-" * 110)
    print(
        f"{'Strategy':<40} {'Annualized Cost':<15} {'Total Interest':<15} "
        f"{'Final Amount':<15} {'Avg Rate':<10} {'Long%':<8}"
    )
    print("-" * 110)

    for i, (strategy_name, data) in enumerate(sorted_results[:top_n]):
        annualized_cost = data['total_cost']
        total_interest = data.get('compound_interest_cost', 0)
        final_amount = data.get('final_amount', 0)
        print(
            f"{i}: {strategy_name:<40} ${annualized_cost:<14,.0f} "
            f"${total_interest:<14,.0f} ${final_amount:<14,.0f} "
            f"{data['avg_rate']:<10.2f} {data['long_term_pct']:<8.1f}"
        )

def print_borrowing_history(result: Dict, max_records: int = 10):
    """
    Print detailed borrowing history for a strategy.

    Args:
        result: Results from CmpBorrow function
        max_records: Maximum number of records to display
    """
    if not result.get('borrowing_history'):
        print("No borrowing history available.")
        return

    print(
        f"\nDetailed Borrowing History "
        f"(showing first {max_records} records):"
    )
    print("-" * 120)
    print(
        "Date         Strategy  Rate%  Duration  Principal    "
        "Interest      New Principal"
    )
    print("-" * 120)

    for record in result['borrowing_history'][:max_records]:
        date_str = record['borrow_date'].strftime('%Y-%m-%d')
        strategy = record['strategy']
        rate = record['actual_rate']
        duration = f"{record['duration_months']}mo"
        principal = f"${record['principal_at_start']:,.0f}"
        interest = f"${record['interest_paid']:,.0f}"
        new_principal = f"${record['principal_at_end']:,.0f}"

        row_str = (
            f"{date_str:<12} {strategy:<8} {rate:<6.2f} {duration:<9} "
            f"{principal:<12} {interest:<12} {new_principal:<15}"
        )
        print(row_str)

    if len(result['borrowing_history']) > max_records:
        remaining = len(result['borrowing_history']) - max_records
        print(f"... and {remaining} more records")

def main():
    """
    Main function to run the treasury backtesting analysis.
    """
    print("Treasury Bill Selection Strategy Backtester")
    print("=" * 50)

    # Load and filter data
    print("Loading and filtering data...")
    try:
        data = load_and_filter_data("yield-curve-rates-1990-2024.csv")
        print(f"Loaded {len(data)} data points (3rd Friday of each month)")

        if data:
            print(f"Date range: {data[0]['Date']} to {data[-1]['Date']}")

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
    results = backtest_strategies(data)

    # Find and display best strategy
    best_strategy_name, best_strategy_data = find_best_strategy(results)

    if best_strategy_data:
        print(f"\nBest Strategy: {best_strategy_name}")
        print(f"Short Term: {best_strategy_data['short_term']}")
        print(f"Long Term: {best_strategy_data['long_term']}")
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
            best_result = CmpBorrow(
                data, best_strategy_data['short_term'],
                best_strategy_data['long_term'],
                best_strategy_data['pick_threshold']
            )
            print_borrowing_history(best_result, max_records=10)
        except Exception as e:
            print(f"Could not display borrowing history: {e}")

    # Print top strategies
    print_results(results, top_n=60)

    # Example of specific strategy comparison
    print("\n" + "=" * 50)
    print("Example: Comparing 1 Mo vs 3 Mo with different thresholds")
    print("=" * 50)

    for threshold in [0.0, 0.1, 0.2, 0.3, 0.5]:
        try:
            result = CmpBorrow(data, '1 Mo', '3 Mo', threshold)
            long_term_pct = result['long_term_choices'] / \
                result['total_periods'] * 100
            print(
                f"Threshold {threshold}: Annualized Cost = "
                f"${result['total_cost']:,.0f}, "
                f"Total Interest = "
                f"${result.get('compound_interest_cost', 0):,.0f}, "
                f"Avg Rate = {result['avg_rate']:.2f}%, "
                f"Long Term % = {long_term_pct:.1f}%"
            )
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")

if __name__ == "__main__":
    main()
