#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
SETUP_LOG="$LOG_DIR/setup.log"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Media Bias Analysis System - Startup Script  ${NC}"
echo -e "${BLUE}================================================${NC}"
echo "Logs will be saved to: $LOG_DIR"

# Function to check command availability
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed. Please install it and try again.${NC}"
        exit 1
    fi
}

# 1. Check Prerequisites
echo -e "\n${GREEN}[1/5] Checking Prerequisites...${NC}"
check_command python3
check_command npm
check_command ollama
echo "All prerequisites found."

# 2. Backend Setup
echo -e "\n${GREEN}[2/5] Setting up Backend...${NC}"
cd backend || { echo -e "${RED}Backend directory not found!${NC}"; exit 1; }

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv >> "$SETUP_LOG" 2>&1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing/Updating Python dependencies (this may take a while)..."
pip install -r requirements.txt >> "$SETUP_LOG" 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install Python dependencies. Check $SETUP_LOG for details.${NC}"
    exit 1
fi

# 3. Model Setup
echo -e "\n${GREEN}[3/5] Checking AI Models...${NC}"
echo "Pulling Ollama model 'granite4:micro-h'..."
ollama pull granite4:micro-h >> "$SETUP_LOG" 2>&1

# 4. Start Backend
echo -e "\n${GREEN}[4/5] Starting Backend Server...${NC}"
nohup uvicorn main:app --reload --port 8000 > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# 5. Frontend Setup & Start
echo -e "\n${GREEN}[5/5] Setting up and Starting Frontend...${NC}"
cd ../react_frontend || { echo -e "${RED}Frontend directory not found!${NC}"; exit 1; }

if [ ! -d "node_modules" ]; then
    echo "Installing Node modules..."
    npm install >> "$SETUP_LOG" 2>&1
fi

echo "Starting Frontend..."
nohup npm run dev > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $BACKEND_PID
    kill $FRONTEND_PID
    echo -e "${GREEN}Services stopped. Goodbye!${NC}"
    exit 0
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo -e "\n${BLUE}================================================${NC}"
echo -e "${GREEN}   System is RUNNING!   ${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Frontend:  http://localhost:5173"
echo -e "Backend:   http://localhost:8000"
echo -e "Logs:      $LOG_DIR"
echo -e "${BLUE}================================================${NC}"
echo "Press Ctrl+C to stop the servers."

# Keep script running
wait
