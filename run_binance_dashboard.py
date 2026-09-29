if __name__ == "__main__":
    import sys
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        from futures_strategy.dashboard_app import run_dashboard
    else:
        from futures_strategy.pair_web import run_dashboard
    run_dashboard()
