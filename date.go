package main

import (
	"fmt"
	"math/big"
	"strconv"
	"strings"
	"time"
	_ "time/tzdata"
)

func isLeapYear(year int) bool {
	return (year%4 == 0 && year%100 != 0) || (year%400 == 0)
}

func daysInMonth(year, month int) int {
	switch month {
	case 1, 3, 5, 7, 8, 10, 12:
		return 31
	case 4, 6, 9, 11:
		return 30
	case 2:
		if isLeapYear(year) {
			return 29
		}
		return 28
	default:
		return 30
	}
}

// civilDay returns the civil day count since 1970-01-01 for any Gregorian date (1..9999).
// Algorithms by Howard Hinnant / Gregorian calendar arithmetic.
func civilDay(y, m, d int) int64 {
	if m <= 2 {
		y -= 1
		m += 9
	} else {
		m -= 3
	}
	era := y / 400
	yoe := y - era*400
	doy := (153*m + 2)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return int64(era)*146097 + int64(doe) - 719468
}

// weekdayFromDate returns the time.Weekday for a year, month, day.
func weekdayFromDate(y, m, d int) time.Weekday {
	t := time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC)
	return t.Weekday()
}

func parseDateWithStore(s string, store *RegisterStore) (int, int, int, error) {
	s = strings.TrimSpace(s)
	// Check for register tag
	if strings.HasPrefix(s, "<|r") && strings.HasSuffix(s, "|>") && (len(s) == 6 || len(s) == 7) {
		numStr := s[3 : len(s)-2]
		idx, err := strconv.Atoi(numStr)
		if err == nil && idx >= 0 && idx < NumRegisters {
			if store == nil {
				return 0, 0, 0, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", idx)
			}
			val, err := store.Get(idx)
			if err != nil {
				return 0, 0, 0, err
			}
			if val.Kind == ValKindDate {
				s = val.DateStr
			} else if val.Kind == ValKindString {
				s = val.Str
			} else {
				return 0, 0, 0, fmt.Errorf("ErrTypeMismatch: expected date string, got %s", val.String())
			}
		}
	}

	s = strings.Trim(strings.TrimSpace(s), `"'`)

	// Strip trailing " (Weekday)" if present (accept own canonical output)
	if idx := strings.Index(s, " ("); idx != -1 && strings.HasSuffix(s, ")") {
		s = s[:idx]
	}

	parts := strings.Split(s, "-")
	if len(parts) != 3 || len(parts[0]) != 4 || len(parts[1]) != 2 || len(parts[2]) != 2 {
		return 0, 0, 0, fmt.Errorf("ErrInvalidDate: expected YYYY-MM-DD format, got %q", s)
	}

	y, errY := strconv.Atoi(parts[0])
	m, errM := strconv.Atoi(parts[1])
	d, errD := strconv.Atoi(parts[2])
	if errY != nil || errM != nil || errD != nil {
		return 0, 0, 0, fmt.Errorf("ErrInvalidDate: expected YYYY-MM-DD format, got %q", s)
	}

	if y < 1 || y > 9999 {
		return 0, 0, 0, fmt.Errorf("ErrInvalidDate: year out of supported range (1..9999): %d", y)
	}
	if m < 1 || m > 12 {
		return 0, 0, 0, fmt.Errorf("ErrInvalidDate: invalid month %d in %q", m, s)
	}
	maxD := daysInMonth(y, m)
	if d < 1 || d > maxD {
		return 0, 0, 0, fmt.Errorf("ErrInvalidDate: invalid day %d for %04d-%02d in %q", d, y, m, s)
	}

	return y, m, d, nil
}

// EvaluateDate evaluates a date tool call against globalRegisters.
func EvaluateDate(payload string) (string, error) {
	val, err := EvaluateDateWithStore(payload, globalRegisters)
	if err != nil {
		return "", err
	}
	return val.String(), nil
}

// EvaluateDateWithStore evaluates a date tool call: add, diff, day_of_week against store.
func EvaluateDateWithStore(payload string, store *RegisterStore) (CalcVal, error) {
	payload = strings.TrimSpace(payload)
	openParen := strings.IndexByte(payload, '(')
	if openParen == -1 || !strings.HasSuffix(payload, ")") {
		return CalcVal{}, fmt.Errorf("ErrSyntax: malformed date call, expected func(args...)")
	}

	fnName := strings.TrimSpace(payload[:openParen])
	argsBody := strings.TrimSpace(payload[openParen+1 : len(payload)-1])

	args, err := parseStrArgsWithStore(argsBody)
	if err != nil {
		return CalcVal{}, err
	}

	resolveDateArg := func(arg StrArg) (int, int, int, error) {
		if arg.IsReg {
			if store == nil {
				return 0, 0, 0, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", arg.RegIdx)
			}
			val, err := store.Get(arg.RegIdx)
			if err != nil {
				return 0, 0, 0, err
			}
			if val.Kind == ValKindDate {
				return parseDateWithStore(val.DateStr, store)
			} else if val.Kind == ValKindString {
				return parseDateWithStore(val.Str, store)
			}
			return 0, 0, 0, fmt.Errorf("ErrTypeMismatch: expected date string, got %s", val.String())
		}
		return parseDateWithStore(arg.RawStr, store)
	}

	resolveIntArg := func(arg StrArg, fn string) (int, error) {
		if arg.IsReg {
			if store == nil {
				return 0, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", arg.RegIdx)
			}
			val, err := store.Get(arg.RegIdx)
			if err != nil {
				return 0, err
			}
			if val.Kind == ValKindRat && val.Rat != nil && val.Rat.IsInt() {
				return int(val.Rat.Num().Int64()), nil
			}
			return 0, fmt.Errorf("ErrTypeMismatch: %s expects integer argument, got %s", fn, val.String())
		}
		n, err := strconv.Atoi(strings.TrimSpace(arg.RawStr))
		if err != nil {
			return 0, fmt.Errorf("ErrSyntax: %s expects integer argument, got %q", fn, arg.RawStr)
		}
		return n, nil
	}

	switch fnName {
	case "add":
		if len(args) != 3 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: add expects 3 arguments: (date, n, unit)")
		}
		y, m, d, err := resolveDateArg(args[0])
		if err != nil {
			return CalcVal{}, err
		}

		n, err := resolveIntArg(args[1], "add")
		if err != nil {
			return CalcVal{}, err
		}

		unit := strings.ToLower(strings.Trim(args[2].RawStr, `"'`))
		var targetY, targetM, targetD int

		switch unit {
		case "days", "day":
			t := time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
			targetY, targetM, targetD = t.Year(), int(t.Month()), t.Day()
		case "weeks", "week":
			t := time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n*7)
			targetY, targetM, targetD = t.Year(), int(t.Month()), t.Day()
		case "months", "month":
			totalMonths := (y*12 + m - 1) + n
			targetY = totalMonths / 12
			targetM = (totalMonths % 12) + 1
			if targetM <= 0 {
				targetM += 12
				targetY--
			}
			maxD := daysInMonth(targetY, targetM)
			targetD = d
			if targetD > maxD {
				targetD = maxD
			}
		case "years", "year":
			targetY = y + n
			targetM = m
			maxD := daysInMonth(targetY, targetM)
			targetD = d
			if targetD > maxD {
				targetD = maxD
			}
		default:
			return CalcVal{}, fmt.Errorf("ErrInvalidUnit: unit must be one of [days, weeks, months, years]")
		}

		if targetY < 1 || targetY > 9999 {
			return CalcVal{}, fmt.Errorf("ErrInvalidDate: resulting year out of range: %d", targetY)
		}

		wd := weekdayFromDate(targetY, targetM, targetD)
		return CalcVal{
			Kind:    ValKindDate,
			DateStr: fmt.Sprintf("%04d-%02d-%02d (%s)", targetY, targetM, targetD, wd.String()),
		}, nil

	case "diff":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: diff expects 2 arguments: (date1, date2)")
		}
		y1, m1, d1, err := resolveDateArg(args[0])
		if err != nil {
			return CalcVal{}, err
		}
		y2, m2, d2, err := resolveDateArg(args[1])
		if err != nil {
			return CalcVal{}, err
		}

		day1 := civilDay(y1, m1, d1)
		day2 := civilDay(y2, m2, d2)
		diffDays := day2 - day1
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(diffDays, 1),
		}, nil

	case "diff_weeks":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: diff_weeks expects 2 arguments: (date1, date2)")
		}
		y1, m1, d1, err := resolveDateArg(args[0])
		if err != nil {
			return CalcVal{}, err
		}
		y2, m2, d2, err := resolveDateArg(args[1])
		if err != nil {
			return CalcVal{}, err
		}

		day1 := civilDay(y1, m1, d1)
		day2 := civilDay(y2, m2, d2)
		diffDays := day2 - day1
		diffWeeks := diffDays / 7
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(diffWeeks, 1),
		}, nil

	case "next_weekday":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: next_weekday expects 2 arguments: (date, weekday)")
		}
		y, m, d, err := resolveDateArg(args[0])
		if err != nil {
			return CalcVal{}, err
		}
		targetWdStr := strings.ToLower(strings.Trim(args[1].RawStr, `"'`))
		weekdayMap := map[string]time.Weekday{
			"sunday": time.Sunday, "sun": time.Sunday,
			"monday": time.Monday, "mon": time.Monday,
			"tuesday": time.Tuesday, "tue": time.Tuesday, "tues": time.Tuesday,
			"wednesday": time.Wednesday, "wed": time.Wednesday,
			"thursday": time.Thursday, "thu": time.Thursday, "thur": time.Thursday, "thurs": time.Thursday,
			"friday": time.Friday, "fri": time.Friday,
			"saturday": time.Saturday, "sat": time.Saturday,
		}
		targetWd, ok := weekdayMap[targetWdStr]
		if !ok {
			return CalcVal{}, fmt.Errorf("ErrSyntax: unknown weekday %q", args[1].RawStr)
		}
		currentWd := weekdayFromDate(y, m, d)
		daysAhead := (int(targetWd) - int(currentWd) + 7) % 7
		if daysAhead == 0 {
			daysAhead = 7
		}
		t := time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC).AddDate(0, 0, daysAhead)
		return CalcVal{
			Kind:    ValKindDate,
			DateStr: fmt.Sprintf("%04d-%02d-%02d (%s)", t.Year(), int(t.Month()), t.Day(), t.Weekday().String()),
		}, nil

	case "quarter":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: quarter expects 1 argument: (date)")
		}
		_, m, _, err := resolveDateArg(args[0])
		if err != nil {
			return CalcVal{}, err
		}
		q := (m-1)/3 + 1
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(int64(q), 1),
		}, nil

	case "days_in_month":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: days_in_month expects 2 arguments: (year, month)")
		}
		year, err := resolveIntArg(args[0], "days_in_month")
		if err != nil {
			return CalcVal{}, err
		}
		monthStr := strings.Trim(strings.TrimSpace(args[1].RawStr), `"'`)
		var month int
		if m, err := strconv.Atoi(monthStr); err == nil {
			month = m
		} else {
			monthMap := map[string]int{
				"january": 1, "jan": 1,
				"february": 2, "feb": 2,
				"march": 3, "mar": 3,
				"april": 4, "apr": 4,
				"may": 5,
				"june": 6, "jun": 6,
				"july": 7, "jul": 7,
				"august": 8, "aug": 8,
				"september": 9, "sep": 9, "sept": 9,
				"october": 10, "oct": 10,
				"november": 11, "nov": 11,
				"december": 12, "dec": 12,
			}
			m, ok := monthMap[strings.ToLower(monthStr)]
			if !ok {
				return CalcVal{}, fmt.Errorf("ErrSyntax: unknown month %q", monthStr)
			}
			month = m
		}
		if month < 1 || month > 12 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: month must be between 1 and 12")
		}
		days := daysInMonth(year, month)
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(int64(days), 1),
		}, nil

	case "is_leap_year":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: is_leap_year expects 1 argument: (year)")
		}
		year, err := resolveIntArg(args[0], "is_leap_year")
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind: ValKindBool,
			Bool: isLeapYear(year),
		}, nil

	case "day_of_week":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: day_of_week expects 1 argument: (date)")
		}
		y, m, d, err := resolveDateArg(args[0])
		if err != nil {
			return CalcVal{}, err
		}
		wd := weekdayFromDate(y, m, d)
		return CalcVal{
			Kind: ValKindString,
			Str:  wd.String(),
		}, nil

	case "now":
		if len(args) > 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: now expects 0 or 1 argument: () or (timezone)")
		}
		loc := time.UTC
		if len(args) == 1 && args[0].RawStr != "" {
			tzStr := strings.Trim(strings.TrimSpace(args[0].RawStr), `"'`)
			if tzStr != "" {
				var err error
				loc, err = time.LoadLocation(tzStr)
				if err != nil {
					return CalcVal{}, fmt.Errorf("ErrSyntax: unknown IANA timezone %q", tzStr)
				}
			}
		}
		nowTime := time.Now()
		if store != nil {
			nowTime = store.GetClock()
		} else if gClock := GetGlobalDefaultClock(); gClock != nil {
			nowTime = *gClock
		}
		t := nowTime.In(loc)
		return CalcVal{
			Kind: ValKindString,
			Str:  t.Format("2006-01-02 15:04 MST"),
		}, nil

	default:
		return CalcVal{}, fmt.Errorf("ErrSyntax: unknown date function %q", fnName)
	}
}
