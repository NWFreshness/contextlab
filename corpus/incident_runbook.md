# Incident Response Runbook

## Resolving Error Code E-4471

When inventory reservations expire, customers see error **E-4471** — "inventory reservation expired."

### Diagnosis Steps

1. Check the reservation timestamp in the logs
2. Verify if the item is still in stock (code **E-4472** = "Item no longer available")
3. Check for race conditions in the reservation system

### Resolution

If the item is still available:
- Re-run the inventory check
- Re-issue reservation with new timestamp
- Contact customer with updated availability

If item is out of stock:
- Check alternative warehouses (code **E-4473** for quantity issues)
- Offer equivalent substitute
- Escalate to inventory team

### Related Codes

| Code | Description | Action |
|------|-------------|--------|
| E-4471 | Reservation expired | Retry reservation |
| E-4472 | Item unavailable | Check alternatives |
| E-4473 | Quantity insufficient | Check stock across warehouses |

### Escalation

If E-4471 persists after retry, escalate to L2 support with:
- Customer ID
- Order ID
- Item SKU
- Timestamp of original error
- All related log entries
