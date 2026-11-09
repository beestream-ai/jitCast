/*
 * Copyright (c) 2026 Beestream Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitcast.util;

import java.util.Locale;

/**
 * Reads jitcast's system-property options. A value that does not parse fails with the property's
 * name instead of falling back to the default ({@link Integer#getInteger} silently does that).
 */
public final class Options {

    private Options() {}

    private static String raw(String name) {
        String v = System.getProperty(name);
        return v == null || v.isBlank() ? null : v.trim();
    }

    private static IllegalArgumentException bad(String name, String value, String expected) {
        return new IllegalArgumentException("-D" + name + "=" + value + ": expected " + expected);
    }

    /** An integer of at least {@code min}. */
    public static int integer(String name, int def, int min) {
        String v = raw(name);
        if (v == null) {
            return def;
        }
        try {
            int i = Integer.parseInt(v);
            if (i >= min) {
                return i;
            }
        } catch (NumberFormatException ignored) {
            // reported below
        }
        throw bad(name, v, "an integer of at least " + min);
    }

    /** A whole number of at least {@code min}. */
    public static long number(String name, long def, long min) {
        String v = raw(name);
        if (v == null) {
            return def;
        }
        try {
            long l = Long.parseLong(v);
            if (l >= min) {
                return l;
            }
        } catch (NumberFormatException ignored) {
            // reported below
        }
        throw bad(name, v, "a whole number of at least " + min);
    }

    /** A byte count of at least {@code min}: a number, optionally followed by KB, MB, GB or TB (powers of 1024). */
    public static long bytes(String name, long def, long min) {
        String v = raw(name);
        if (v == null) {
            return def;
        }
        long b = parseBytes(v);
        if (b < min) {
            throw bad(name, v, "a size of at least " + min + " bytes, such as 512MB or 8GB");
        }
        return b;
    }

    /** {@code text} as a byte count, or -1 when it is not one. */
    public static long parseBytes(String text) {
        String s = text.trim().toUpperCase(Locale.ROOT);
        long unit = 1;
        for (String[] u : new String[][] {{"TB", "1099511627776"}, {"GB", "1073741824"}, {"MB", "1048576"}, {"KB", "1024"}, {"B", "1"}}) {
            if (s.endsWith(u[0])) {
                unit = Long.parseLong(u[1]);
                s = s.substring(0, s.length() - u[0].length()).trim();
                break;
            }
        }
        try {
            return Math.multiplyExact(Long.parseLong(s), unit);
        } catch (NumberFormatException | ArithmeticException e) {
            return -1;
        }
    }

    /** {@code true} or {@code false}, in any case. */
    public static boolean bool(String name, boolean def) {
        String v = raw(name);
        if (v == null) {
            return def;
        }
        if (v.equalsIgnoreCase("true")) {
            return true;
        }
        if (v.equalsIgnoreCase("false")) {
            return false;
        }
        throw bad(name, v, "true or false");
    }

    /** One of {@code allowed} (lower case), compared ignoring case. */
    public static String choice(String name, String def, String... allowed) {
        String v = raw(name);
        if (v == null) {
            return def;
        }
        for (String a : allowed) {
            if (a.equalsIgnoreCase(v)) {
                return a;
            }
        }
        throw bad(name, v, "one of " + String.join(", ", allowed));
    }

    /** A constant of {@code type}, written in lower case with dashes ({@code cublas-fp16} for {@code CUBLAS_FP16}). */
    public static <E extends Enum<E>> E enumeration(String name, E def, Class<E> type) {
        String v = raw(name);
        return v == null ? def : parseEnum(name, v, type);
    }

    /** {@code value} as a constant of {@code type}, reported as {@code -D<name>} when it is not one. */
    public static <E extends Enum<E>> E parseEnum(String name, String value, Class<E> type) {
        String key = value.trim().toUpperCase(Locale.ROOT).replace('-', '_');
        for (E e : type.getEnumConstants()) {
            if (e.name().equals(key)) {
                return e;
            }
        }
        StringBuilder names = new StringBuilder();
        for (E e : type.getEnumConstants()) {
            names.append(names.length() == 0 ? "" : ", ").append(e.name().toLowerCase(Locale.ROOT).replace('_', '-'));
        }
        throw bad(name, value, "one of " + names);
    }
}
