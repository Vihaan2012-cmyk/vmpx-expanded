vmpx-vm
=======

The VM analysis engine: entry discovery, bootstrap and handler-codec recovery,
stateful decoding, symbolic execution, native-exit and continuation analysis, OEP
decision, input-domain inference and anti-debug neutralization.
*Depends on:* ``vmpx-pe``, ``vmpx-disasm``, ``iced-x86``.

This is the largest crate (about 26,000 lines). The pipeline stages it implements
are described in :doc:`/tool/pipeline/vm-recovery`,
:doc:`/tool/pipeline/symbolic-execution`, :doc:`/tool/pipeline/oep-recovery` and
:doc:`/tool/pipeline/function-recovery`.

Discovery
---------

``discover_vm_entries``, ``analyze_vm_entry_at``, ``VmDiscoveryLimits``,
``VmDiscoveryReport``, ``VmEntryCandidate``, ``VmEntrySource``,
``VmEntryReachability``, ``VmEntryEvidence``, ``VmInstance`` and
``VmDiscoveryError``.

Bootstrap and profile
---------------------

``recover_bootstrap_dispatch``, ``recover_entry_profile``,
``recover_entry_paths``, ``RecoveredEntryProfile``, ``RecoveredEntryPath``,
``RecoveredEntryPathFamily`` and ``EntryPathRecoveryLimits``.

``RecoveredEntryPathFamily`` carries ``domain_complete`` and
``all_paths_reach_vm_exit``, the two facts the OEP decision and the coverage gate
depend on.

Handler analysis
----------------

``backward_slice_handler_trace`` (``HandlerSliceConfig``, ``NativeBackwardSlice``),
``extract_relative_dispatch_codecs`` (with ``_with_postlude`` and
``_with_postlude_direction`` variants), ``extract_keyed_operand_codecs`` (and
``_direction``), ``extract_handler_effects``, ``classify_handler_effects``,
``prove_handler_value_flow``, ``recover_handler_codecs``,
``resolve_vm_profile_roles`` and ``analyze_vm_transition``.

Decoding
--------

``decode_linear_vm_stream`` (and ``_direction``) and
``recover_and_decode_linear_vm_stream``. The codec types are
``KeyedTransformCodec``, ``RelativeDispatchCodec``, ``RelativeThreadedCodec`` and
``RelativeDispatchState``; the output types are ``DecodedVmInstruction``,
``VmHandlerDecodeSpec``, ``LinearVmDecodeReport`` and ``LinearDecodeStop``. The
transform vocabulary is ``Transform``, ``TransformChain``, ``TransformOp`` and
``BitWidth``.

Model
-----

``VmHandlerSemantic``, ``VmHandlerAnalysis``, ``VmContextLayout``,
``VmDecodeState``, ``StateValue``, ``VmInstruction``, ``BytecodeDirection``,
``RecoveryStatus`` and ``EffectKnowledge``.

Symbolic execution
------------------

``execute_symbolic_linear_block`` (and ``_with_initial_state``),
``execute_symbolic_native_call_boundary_with_initial_state``,
``enumerate_symbolic_value``, ``explore_vm_profile_chain``,
``explore_vm_profile_paths``, ``assemble_profile_path_graph``,
``concretize_with_immutable_image`` and ``recover_modulo_image_targets``.

The expression IR is ``Expr``, ``ExprKind``, ``BinaryOp``, ``UnaryOp`` and
``ResizeOp``.

Native boundary
---------------

``analyze_vm_exit``, ``analyze_x64_native_frame``, ``map_native_context_scatter``,
``recover_native_exit_state``, ``recover_profile_chain_native_exit_state``,
``continue_through_native_vm_entry``, ``analyze_native_call_handler``,
``apply_native_call_effect``, ``resolve_native_call_import``,
``summarize_known_import`` and ``analyze_native_continuation``.

Exceptions
----------

``analyze_exception_stack_sled`` and ``analyze_vm_memory_exception_coverage``
(and ``_with_state``).

Checksum handlers
-----------------

``analyze_table_checksum_handler``, ``TableChecksumMetadata``,
``TableChecksumMetadataError``, ``VmSemanticMetadata``,
``TableChecksumHandlerReport`` and ``RecoveredTableChecksumHandler``.

OEP, input-closure and input domains (SP2)
------------------------------------------

.. code-block:: rust

    pub fn decide_oep(
        paths: &[Option<Rva>],
        family_complete: bool,
        is_native: impl Fn(Rva) -> bool,
    ) -> OepDecision;

    pub fn is_expr_input_closed(expr: &Expr, allowed: &BTreeSet<String>) -> bool;
    pub fn is_input_closed(exit: &NativeExitState, allowed: &BTreeSet<String>) -> bool;

    pub fn add_flags_cover(
        bits: u32,
        observations: &[AddFlagsObservation],
        limits: InputDomainLimits,
    ) -> Option<Vec<u64>>;
    pub fn infer_branch_input_domains(
        target: &Expr,
        limits: InputDomainLimits,
    ) -> Result<InferredInputDomains, InputDomainRefusal>;

``InputDomainLimits`` defaults to ``max_values_per_variable = 4096`` and
``max_exhaustive_bits = 8``. ``DomainCertificate`` is ``Exhaustive { bits }`` or
``AddFlagsCover { bits, observations }``. ``InputDomainRefusal`` is
``MemoryDependent``, ``Unsupported { register, reason }`` or
``TooManyValues { register, limit }``.

Anti-debug neutralization (SP5)
-------------------------------

``neutralize(guards, imports, opts)`` (returns the rewritten guards and the
``NeutralizeAction`` list), ``NeutralizeOptions``, ``NeutralizeAction``,
``NeutralizeConstruct``, ``RecoveredGuard`` and ``DetectedWhen``, plus the
constants ``INTEGRITY_CHECKSUM_INTRINSIC`` and
``TIMESTAMP_COUNTER_INTRINSIC_PREFIX``. See :doc:`/tool/pipeline/verification`.

Module map
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``bootstrap``, ``native_context``
     - Bootstrap recovery, register roles, context scatter
   * - ``entry_profile``, ``entry_paths``
     - Entry profile and finite-domain path family recovery
   * - ``dispatch_extract``, ``handler_slice``
     - Dispatch codec extraction and backward slices
   * - ``decoder``, ``transform``, ``codec_recovery``
     - The stateful decoder and reversible transforms
   * - ``operand_extract``, ``effect_extract``, ``value_flow``, ``model``
     - Operands, effects, def-use proof and the semantic model
   * - ``semantic_exec``, ``expr``, ``symbolic_domain``
     - Symbolic executor and expression IR
   * - ``profile_explore``, ``profile_paths``, ``path_graph``
     - Chaining blocks and enumerating input domains
   * - ``image_concretize``
     - Image-backed load concretization
   * - ``native_call``, ``native_call_verify``, ``call_effects``, ``native_api``
     - Native-call bridge and import summaries
   * - ``vm_exit``, ``native_exit_state``, ``native_vm_chain``, ``native_continuation``
     - Exit proof, exit state, VM-to-VM chaining, native continuation
   * - ``transition_extract``
     - Native and VM transition extraction
   * - ``exception_flow``
     - Exception-driven control flow
   * - ``checksum``, ``checksum_handler``
     - Integrity handler recovery
   * - ``oep``, ``input_closed``, ``input_domain``
     - OEP decision and input analysis
   * - ``neutralize``
     - Anti-debug neutralization transform
