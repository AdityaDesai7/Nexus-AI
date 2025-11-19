# models/database.py
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd


class Database:
    """Handle all database operations"""

    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                initial_capital REAL DEFAULT 1000000.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Portfolio table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                total_invested REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, ticker)
            )
        """)

        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                total_amount REAL NOT NULL,
                agent_recommendation TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # User balance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_balance (
                user_id INTEGER PRIMARY KEY,
                available_cash REAL NOT NULL,
                invested_amount REAL DEFAULT 0.0,
                total_portfolio_value REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        conn.commit()
        conn.close()

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username: str, email: str, password: str) -> tuple[bool, str]:
        """Register a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            password_hash = self.hash_password(password)
            initial_capital = 1000000.0

            cursor.execute("""
                INSERT INTO users (username, email, password_hash, initial_capital)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, initial_capital))

            user_id = cursor.lastrowid

            # Initialize user balance
            cursor.execute("""
                INSERT INTO user_balance (user_id, available_cash)
                VALUES (?, ?)
            """, (user_id, initial_capital))

            conn.commit()
            conn.close()

            return True, "Registration successful! You have been credited with ₹10,00,000 virtual capital."

        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return False, "Username already exists. Please login or choose a different username."
            elif "email" in str(e):
                return False, "Email already registered. Please login."
            return False, "Registration failed. Please try again."
        except Exception as e:
            return False, f"Error: {str(e)}"

    def login_user(self, username: str, password: str) -> tuple[bool, Optional[int], str]:
        """Login user and return user_id"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            password_hash = self.hash_password(password)

            cursor.execute("""
                SELECT user_id, username FROM users
                WHERE username = ? AND password_hash = ?
            """, (username, password_hash))

            result = cursor.fetchone()
            conn.close()

            if result:
                return True, result[0], f"Welcome back, {result[1]}!"
            else:
                return False, None, "Invalid username or password."

        except Exception as e:
            return False, None, f"Login error: {str(e)}"

    def is_admin(self, username: str) -> bool:
        """Check if user is admin"""
        # Hardcoded admin usernames (you can change these)
        ADMIN_USERS = ['admin', 'aditya_admin', 'superadmin']
        return username.lower() in ADMIN_USERS

    def get_user_balance(self, user_id: int) -> Dict:
        """Get user's current balance and portfolio value"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT available_cash, invested_amount, total_portfolio_value
            FROM user_balance
            WHERE user_id = ?
        """, (user_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'available_cash': result[0],
                'invested_amount': result[1],
                'total_portfolio_value': result[2]
            }
        return {'available_cash': 0, 'invested_amount': 0, 'total_portfolio_value': 0}

    def get_portfolio(self, user_id: int) -> pd.DataFrame:
        """Get user's portfolio"""
        conn = self.get_connection()

        query = """
            SELECT ticker, quantity, avg_price, total_invested, last_updated
            FROM portfolio
            WHERE user_id = ? AND quantity > 0
            ORDER BY last_updated DESC
        """

        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()

        return df

    def execute_trade(self, user_id: int, ticker: str, action: str,
                      quantity: int, price: float, agent_rec: str = None) -> tuple[bool, str]:
        """Execute a trade (BUY or SELL)"""
        try:
            print(f"[DEBUG] execute_trade called with user_id={user_id}, ticker={ticker}, action={action}")

            conn = self.get_connection()
            cursor = conn.cursor()

            # Get current balance
            balance = self.get_user_balance(user_id)
            available_cash = balance['available_cash']

            total_amount = quantity * price

            print(f"[DEBUG] Available cash: {available_cash}, Total amount: {total_amount}")

            if action == "BUY":
                # Check if user has enough cash
                if available_cash < total_amount:
                    return False, f"Insufficient funds! Available: ₹{available_cash:,.2f}, Required: ₹{total_amount:,.2f}"

                # Deduct cash
                new_cash = available_cash - total_amount

                # Update or insert portfolio
                cursor.execute("""
                    SELECT quantity, avg_price, total_invested
                    FROM portfolio
                    WHERE user_id = ? AND ticker = ?
                """, (user_id, ticker))

                existing = cursor.fetchone()

                if existing:
                    # Update existing position
                    old_qty, old_avg_price, old_invested = existing
                    new_qty = old_qty + quantity
                    new_invested = old_invested + total_amount
                    new_avg_price = new_invested / new_qty

                    cursor.execute("""
                        UPDATE portfolio
                        SET quantity = ?, avg_price = ?, total_invested = ?,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND ticker = ?
                    """, (new_qty, new_avg_price, new_invested, user_id, ticker))
                else:
                    # Create new position
                    cursor.execute("""
                        INSERT INTO portfolio (user_id, ticker, quantity, avg_price, total_invested)
                        VALUES (?, ?, ?, ?, ?)
                    """, (user_id, ticker, quantity, price, total_amount))

                # Update balance
                cursor.execute("""
                    UPDATE user_balance
                    SET available_cash = ?,
                        invested_amount = invested_amount + ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (new_cash, total_amount, user_id))

                message = f"✅ BUY order executed: {quantity} shares of {ticker} at ₹{price:.2f}"

            elif action == "SELL":
                # Check if user has the stock
                cursor.execute("""
                    SELECT quantity, avg_price, total_invested
                    FROM portfolio
                    WHERE user_id = ? AND ticker = ?
                """, (user_id, ticker))

                existing = cursor.fetchone()

                if not existing or existing[0] < quantity:
                    available_qty = existing[0] if existing else 0
                    return False, f"Insufficient shares! You have {available_qty} shares of {ticker}"

                old_qty, old_avg_price, old_invested = existing
                new_qty = old_qty - quantity

                # Calculate amount to return
                amount_to_return = quantity * price
                invested_to_reduce = (quantity / old_qty) * old_invested

                # Add cash back
                new_cash = available_cash + amount_to_return

                if new_qty > 0:
                    # Update portfolio
                    new_invested = old_invested - invested_to_reduce
                    cursor.execute("""
                        UPDATE portfolio
                        SET quantity = ?, total_invested = ?,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND ticker = ?
                    """, (new_qty, new_invested, user_id, ticker))
                else:
                    # Remove from portfolio
                    cursor.execute("""
                        DELETE FROM portfolio
                        WHERE user_id = ? AND ticker = ?
                    """, (user_id, ticker))

                # Update balance
                cursor.execute("""
                    UPDATE user_balance
                    SET available_cash = ?,
                        invested_amount = invested_amount - ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (new_cash, invested_to_reduce, user_id))

                profit_loss = amount_to_return - invested_to_reduce
                pl_text = f"Profit: ₹{profit_loss:,.2f}" if profit_loss > 0 else f"Loss: ₹{abs(profit_loss):,.2f}"
                message = f"✅ SELL order executed: {quantity} shares of {ticker} at ₹{price:.2f} ({pl_text})"

            else:
                return False, "Invalid action. Must be BUY or SELL."

            # Record transaction
            cursor.execute("""
                INSERT INTO transactions (user_id, ticker, action, quantity, price, total_amount, agent_recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, ticker, action, quantity, price, total_amount, agent_rec))

            conn.commit()
            conn.close()

            print(f"[DEBUG] Trade executed successfully!")
            return True, message

        except Exception as e:
            print(f"[DEBUG] Trade execution error: {str(e)}")
            return False, f"Trade execution error: {str(e)}"

    def get_transaction_history(self, user_id: int, limit: int = 50) -> pd.DataFrame:
        """Get user's transaction history"""
        conn = self.get_connection()

        query = """
            SELECT transaction_id, ticker, action, quantity, price, 
                   total_amount, agent_recommendation, timestamp
            FROM transactions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(user_id, limit))
        conn.close()

        return df

    def update_portfolio_values(self, user_id: int, current_prices: Dict[str, float]):
        """Update portfolio with current market prices"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get portfolio
        cursor.execute("""
            SELECT ticker, quantity, total_invested
            FROM portfolio
            WHERE user_id = ?
        """, (user_id,))

        holdings = cursor.fetchall()
        total_current_value = 0

        for ticker, quantity, invested in holdings:
            current_price = current_prices.get(ticker, 0)
            total_current_value += quantity * current_price

        # Update user balance
        cursor.execute("""
            UPDATE user_balance
            SET total_portfolio_value = ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (total_current_value, user_id))

        conn.commit()
        conn.close()

    def get_user_stats(self, user_id: int) -> Dict:
        """Get comprehensive user statistics"""
        balance = self.get_user_balance(user_id)
        portfolio = self.get_portfolio(user_id)

        total_value = balance['available_cash'] + balance['total_portfolio_value']
        initial_capital = 1000000.0
        profit_loss = total_value - initial_capital
        profit_loss_pct = (profit_loss / initial_capital) * 100

        return {
            'total_value': total_value,
            'available_cash': balance['available_cash'],
            'invested_amount': balance['invested_amount'],
            'portfolio_value': balance['total_portfolio_value'],
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'num_holdings': len(portfolio)
        }

    # ========================================
    # ADMIN FUNCTIONS
    # ========================================

    def get_all_users(self) -> pd.DataFrame:
        """Get all registered users (Admin only)"""
        conn = self.get_connection()

        query = """
            SELECT 
                u.user_id,
                u.username,
                u.email,
                u.initial_capital,
                u.created_at,
                ub.available_cash,
                ub.invested_amount,
                ub.total_portfolio_value,
                (ub.available_cash + ub.total_portfolio_value) as total_value
            FROM users u
            LEFT JOIN user_balance ub ON u.user_id = ub.user_id
            ORDER BY u.created_at DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_user_portfolio_admin(self, user_id: int) -> Dict:
        """Get detailed user portfolio (Admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get user info
        cursor.execute("""
            SELECT username, email, created_at, initial_capital
            FROM users WHERE user_id = ?
        """, (user_id,))
        user_info = cursor.fetchone()

        # Get portfolio
        portfolio_df = self.get_portfolio(user_id)

        # Get transaction count
        cursor.execute("""
            SELECT COUNT(*) FROM transactions WHERE user_id = ?
        """, (user_id,))
        transaction_count = cursor.fetchone()[0]

        # Get balance
        balance = self.get_user_balance(user_id)

        conn.close()

        if user_info:
            return {
                'username': user_info[0],
                'email': user_info[1],
                'created_at': user_info[2],
                'initial_capital': user_info[3],
                'portfolio': portfolio_df,
                'transaction_count': transaction_count,
                'balance': balance
            }
        return None

    def get_all_transactions_admin(self, limit: int = 100) -> pd.DataFrame:
        """Get all transactions across all users (Admin only)"""
        conn = self.get_connection()

        query = """
            SELECT 
                t.transaction_id,
                u.username,
                t.ticker,
                t.action,
                t.quantity,
                t.price,
                t.total_amount,
                t.agent_recommendation,
                t.timestamp
            FROM transactions t
            JOIN users u ON t.user_id = u.user_id
            ORDER BY t.timestamp DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        return df

    def get_system_stats(self) -> Dict:
        """Get overall system statistics (Admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        # Total transactions
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = cursor.fetchone()[0]

        # Total capital in system
        cursor.execute("""
            SELECT SUM(available_cash + total_portfolio_value) 
            FROM user_balance
        """)
        total_capital = cursor.fetchone()[0] or 0

        # Total invested
        cursor.execute("SELECT SUM(invested_amount) FROM user_balance")
        total_invested = cursor.fetchone()[0] or 0

        # Most traded stocks
        cursor.execute("""
            SELECT ticker, COUNT(*) as trade_count
            FROM transactions
            GROUP BY ticker
            ORDER BY trade_count DESC
            LIMIT 5
        """)
        top_stocks = cursor.fetchall()

        # Active users (users with transactions)
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id) FROM transactions
        """)
        active_users = cursor.fetchone()[0]

        # Total buy vs sell
        cursor.execute("""
            SELECT action, COUNT(*) as count
            FROM transactions
            GROUP BY action
        """)
        buy_sell_stats = cursor.fetchall()

        conn.close()

        return {
            'total_users': total_users,
            'active_users': active_users,
            'total_transactions': total_transactions,
            'total_capital': total_capital,
            'total_invested': total_invested,
            'top_stocks': top_stocks,
            'buy_sell_stats': buy_sell_stats
        }

    def get_user_leaderboard(self) -> pd.DataFrame:
        """Get user leaderboard by P&L (Admin only)"""
        conn = self.get_connection()

        query = """
            SELECT 
                u.username,
                u.initial_capital,
                (ub.available_cash + ub.total_portfolio_value) as total_value,
                (ub.available_cash + ub.total_portfolio_value - u.initial_capital) as profit_loss,
                ((ub.available_cash + ub.total_portfolio_value - u.initial_capital) / u.initial_capital * 100) as profit_loss_pct,
                (SELECT COUNT(*) FROM transactions WHERE user_id = u.user_id) as total_trades
            FROM users u
            JOIN user_balance ub ON u.user_id = ub.user_id
            ORDER BY profit_loss DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df


# Global database instance
_db = None


def get_database() -> Database:
    """Get global database instance"""
    global _db
    if _db is None:
        _db = Database()
    return _db
