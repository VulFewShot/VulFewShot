static void FUN1(VAR1 *VAR2, VAR3 *VAR4, const char *VAR5,
                                 void *VAR6, VAR7 **VAR8)
{
    VAR9 *VAR10 = FUN2(VAR2);
    VAR11 *VAR12 = VAR6;
    VAR13 *VAR14 = FUN3(VAR10, VAR12);
    char VAR15[] = "";
    char *VAR16 = VAR15;
    int VAR17 = 0;
    VAR17 = snprintf(VAR15, sizeof(VAR15), "",
                  VAR14->VAR18, VAR14->VAR19, VAR14->VAR20, VAR14->VAR21);
    assert(VAR17 == sizeof(VAR15) - 1);
    FUN4(VAR4, VAR5, &VAR16, VAR8);
}