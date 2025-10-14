#!/bin/bash
#
# Update all worker constitutions with governance and auto-sync rules
#

echo "Updating all worker constitutions with governance rules..."

for WORKER_ID in {1..8}; do
    CONST_FILE="/home/diddy/Desktop/PRISM-Worker-${WORKER_ID}/.worker-vault/Constitution/WORKER_${WORKER_ID}_CONSTITUTION.md"

    if [ ! -f "$CONST_FILE" ]; then
        echo "⚠️  Worker $WORKER_ID constitution not found - skipping"
        continue
    fi

    # Check if already updated
    if grep -q "Article V: Governance Enforcement" "$CONST_FILE"; then
        echo "✅ Worker $WORKER_ID constitution already updated"
        continue
    fi

    echo "📝 Updating Worker $WORKER_ID constitution..."

    # Append governance rules
    cat >> "$CONST_FILE" <<EOF

## Article V: Governance Enforcement

BEFORE ALL WORK:
- Governance engine runs automatically via worker_start.sh
- Checks file ownership, dependencies, build hygiene
- **VIOLATION = IMMEDIATE BLOCKING** until resolved

YOU SHALL:
- Accept governance verdicts
- Fix violations immediately
- Not bypass governance checks
- Report issues to Worker 0-Alpha if governance incorrect

GOVERNANCE RULES:
1. ✅ Only edit files you own
2. ✅ Have required dependencies before proceeding
3. ✅ Code must build before committing
4. ✅ Use GPU for all compute
5. ✅ Commit daily with proper messages
6. ✅ Follow integration protocol
7. ✅ Use auto-sync system

## Article VI: Auto-Sync System

YOU SHALL:
- Use \`./worker_start.sh $WORKER_ID\` to begin each session
- Allow automatic dependency pulling
- Wait gracefully when dependencies not ready
- Work on alternative features when blocked

YOU SHALL NOT:
- Manually track dependencies
- Skip auto-sync checks
- Proceed when governance blocks
- Bypass automatic integration

## Article VII: Deliverable Publishing

WHEN FEATURES COMPLETE:
- Publish to deliverables branch immediately
- Update .worker-deliverables.log
- Update DELIVERABLES.md status
- Notify dependent workers

WORKER $WORKER_ID SPECIFIC:
$(case $WORKER_ID in
    1) echo "- **Week 3**: Time series module MUST be published (CRITICAL - unblocks Workers 5 & 7)" ;;
    2) echo "- **Week 2**: Time series kernels published (CRITICAL - unblocks Worker 1)"
       echo "- **Week 3**: Pixel kernels published (CRITICAL - unblocks Worker 3)" ;;
    3) echo "- **Week 4**: Pixel processing completed"
       echo "- **Week 5-6**: PWSA/Finance apps complete" ;;
    4) echo "- **Week 5-6**: Telecom/Robotics apps complete" ;;
    5) echo "- **Week 4-5**: Advanced thermodynamic features complete" ;;
    6) echo "- **Week 5-6**: Advanced LLM features complete" ;;
    7) echo "- **Week 5-6**: Drug discovery and robotics complete" ;;
    8) echo "- **Week 6-7**: API server, deployment, and documentation complete" ;;
esac)
EOF

    echo "✅ Worker $WORKER_ID constitution updated"
done

echo ""
echo "✅ All worker constitutions updated with governance rules!"
