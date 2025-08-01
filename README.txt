Treasury Bill Selection Strategy Backtester
==========================================

This program analyzes different treasury bill selection strategies using historical yield curve data.

Common Usage Patterns
====================

1. Generate Detailed History for All Strategies
   -------------------------------------------
   Use -s -1 to generate detailed borrowing history for all strategies:
   
   python pick-treasury.py -s -1
   
   This will create CSV files in the results/ directory for each strategy,
   showing the complete borrowing history with dates, rates, and amounts.

2. Run Strategy Stability Test
   ---------------------------
   Use -t -y 5 to run a stability test with 5-year rolling periods:
   
   python pick-treasury.py -t -y 5
   
   This tests each strategy's performance over rolling 5-year periods and
   calculates mean rank, standard deviation, and stability metrics.

3. Run Stability Test with Different Period Lengths
   -----------------------------------------------
   You can vary the rolling period length:
   
   python pick-treasury.py -t -y 3    # 3-year periods
   python pick-treasury.py -t -y 10   # 10-year periods

4. Generate History for Specific Strategy
   -------------------------------------
   Use -s followed by strategy ID to generate detailed history for one strategy:
   
   python pick-treasury.py -s 80      # Generate history for strategy ID 80

5. Combine Stability Test with History Generation
   --------------------------------------------
   You can run both tests together:
   
   python pick-treasury.py -t -y 5 -s -1

Command Line Arguments
=====================

-s, --strategy-id STRATEGY_ID
    Generate detailed borrowing history for specific strategy ID
    Use -1 for all strategies

-t, --stability-test
    Run strategy stability test with rolling N-year periods

-y, --stability-years YEARS
    Number of years for stability test rolling periods (default: 5)

-o, --output-file OUTPUT_FILE
    Output file for borrowing history (default: borrow-history.csv)

-f, --filtered-data-file FILTERED_DATA_FILE
    Output file for filtered monthly yield data (default: monthly-yield.csv)

-k, --skip-records SKIP_RECORDS
    Number of records to skip from the beginning when printing
    detailed history (default: 1000000)

Example Output
=============

Stability Test Results:
Strategy Name                                      Mean   StDev  Min  Max  Periods 
------------------------------------------------------------------------------------------------------------
fixed_1Mo                                          5.7    8.6    1    52   217     
1Mo_3Mo_thresh_1.0                                 6.5    8.7    1    53   217     
1Mo_3Mo_thresh_0.5                                 7.7    9.1    1    55   217     

The results show:
- Strategy Name: The strategy identifier
- Mean: Average rank across all periods (lower is better)
- StDev: Standard deviation of ranks (lower = more stable)
- Min/Max: Best and worst ranks achieved
- Periods: Number of periods the strategy was tested in

Data Requirements
================

The program requires a CSV file named "yield-curve-rates-1990-2024.csv" with columns:
- Date (MM/DD/YY format)
- 1 Mo, 3 Mo, 6 Mo, 1 Yr, 2 Yr (treasury rates)

The program filters for 3rd Friday of each month and uses the complete data range
where all treasury periods have available data. 