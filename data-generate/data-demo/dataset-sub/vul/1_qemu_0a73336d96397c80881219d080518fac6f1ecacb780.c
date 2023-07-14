static int FUN1(void *VAR1,
                                  const char *VAR2, const char *VAR3,
                                  VAR4 **VAR5)
{
    VAR6 *VAR7 = VAR1;
    if (strcmp(VAR2, "") == 0 && strcmp(VAR3, "") == 0) {
        VAR7->VAR8 = true;
    } else if (strcmp(VAR2, "") == 0) {
        VAR7->VAR9 = true;
    } else if (strcmp(VAR2, "") == 0) {
    } else {
        FUN2(VAR5,
                   "",
                   VAR2, VAR3);
        return -1;
    }
    return 0;
}